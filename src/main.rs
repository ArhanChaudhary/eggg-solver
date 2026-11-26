use clap::{Args, Parser, Subcommand, ValueEnum};
use eggg_solver::cube::{Cube, Facelets};
use eggg_solver::notation::{Style, format_moves, parse_moves};
use eggg_solver::solver::{
    JointTable, Options, P1Extra, Solution, Solver, Tables, load_tables_with, verify,
};
use rand::SeedableRng;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(
    name = "eggg-solver",
    version,
    about = "Fast three-phase 4x4x4 solver (outer block turn metric)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Solve positions given as scrambles or facelet strings.
    Solve(SolveArgs),
    /// Generate (or check) the pruning tables.
    Gen(GenArgs),
    /// Benchmark on random positions.
    Bench(BenchArgs),
    /// Print the facelets of positions.
    Show(ShowArgs),
    /// Benchmark the phase-3 search alone: optimal solves of random 3x3x3 positions.
    #[command(hide = true)]
    Bench3(Bench3Args),
}

#[derive(Args)]
struct Bench3Args {
    #[arg(short = 'N', long, default_value_t = 10)]
    count: usize,
    #[arg(long, default_value_t = 2026)]
    seed: u64,
    #[command(flatten)]
    tables: TableArgs,
    #[arg(short = 'j', long)]
    threads: Option<usize>,
}

#[derive(Args, Clone)]
#[command(next_help_heading = "Input")]
struct InputArgs {
    /// Scrambles (WCA or SiGN notation) or 96-character facelet strings (faces U R F D L B,
    /// each row by row). None: read one position per line from stdin ('#' starts a comment).
    #[arg(value_name = "SCRAMBLE|FACELETS")]
    position: Vec<String>,
    /// Solve N uniformly random positions instead.
    #[arg(long, value_name = "N", conflicts_with = "position")]
    random: Option<usize>,
    /// Seed for --random.
    #[arg(long, default_value_t = 1, requires = "random")]
    seed: u64,
}

#[derive(Args, Clone)]
#[command(next_help_heading = "Limits")]
struct LimitArgs {
    /// Search for at most this many seconds per position (default: `solve` runs until it is
    /// stopped or `-l` is reached, printing each improvement; `bench` 5 seconds).
    #[arg(short = 't', long, value_name = "SECONDS")]
    time: Option<f64>,
    /// Stop as soon as a solution of at most this many moves is found.
    #[arg(short = 'l', long, value_name = "MOVES")]
    max_length: Option<usize>,
    /// Give up after this many seconds even without a solution (by default the search goes on
    /// until it has one; `bench` defaults to 4 times the time limit).
    #[arg(long, value_name = "SECONDS")]
    hard_limit: Option<f64>,
}

/// Table sets by memory use (the joint phase-2 table pays off before a larger phase-3 table).
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Tier {
    /// The largest tier that uses at most a third of the physical memory (xl and xxl only
    /// once their phase-3 table exists: generating it takes ~20 minutes).
    Auto,
    /// ~0.16 GB: h48 h0 (slow: phase 2 has only the pairing table).
    Tiny,
    /// ~0.35 GB: h48 h3.
    Small,
    /// ~2.3 GB: h48 h5 and the joint phase-2 table.
    Medium,
    /// ~5.13 GB: h48 h6, the joint phase-2 table, the phase-1 move masks and wing paths up to
    /// distance 6, with the reduced lists of distance 7 (0.42 GB).
    Large,
    /// ~9.5 GB: h48 h7, the exact joint phase-2 table, the phase-1 move masks and wing paths
    /// up to distance 7 (1.8 GB).
    Huge,
    /// ~20.9 GB: h48 h9, the exact joint phase-2 table, the phase-1 move masks and wing paths
    /// up to distance 7 (phase 3 3.6x faster than huge; single-threaded first solutions ~4x
    /// sooner).
    Xl,
    /// ~66.5 GB: h48 h11 (the largest), the exact joint phase-2 table, the phase-1 move masks
    /// and wing paths up to distance 7 (phase 3 11x faster than huge; single-threaded first
    /// solutions ~12x sooner).
    Xxl,
}

impl Tier {
    /// (h48 K, joint phase-2 table, phase-1 move masks, wing-path distance, peak memory in
    /// bytes: measured, plus the wing-path table).
    fn tables(self) -> (u32, JointTable, bool, u8, f64) {
        use JointTable::{Exact, Mod3, None};
        match self {
            Tier::Tiny => (0, None, false, 0, 0.16e9),
            Tier::Small => (3, None, false, 0, 0.35e9),
            Tier::Medium => (5, Mod3, false, 0, 2.3e9),
            Tier::Large => (6, Mod3, true, 6, 5.13e9),
            Tier::Huge | Tier::Auto => (7, Exact, true, 7, 9.5e9),
            Tier::Xl => (9, Exact, true, 7, 20.9e9),
            Tier::Xxl => (11, Exact, true, 7, 66.5e9),
        }
    }

    fn resolve(self, dir: &std::path::Path) -> Tier {
        if self != Tier::Auto {
            return self;
        }
        // The xl and xxl phase-3 tables take ~20 minutes to generate: only used when present.
        let present = |t: Tier| {
            let k = t.tables().0 as usize;
            let base = eggg_solver::h48::DEFAULT_BASE[k];
            dir.join(format!("p3-h48-h{k}b{base}.tab")).exists()
        };
        // SAFETY: sysconf has no preconditions.
        let ram = unsafe {
            libc::sysconf(libc::_SC_PHYS_PAGES) as f64 * libc::sysconf(libc::_SC_PAGESIZE) as f64
        };
        [
            Tier::Xxl,
            Tier::Xl,
            Tier::Huge,
            Tier::Large,
            Tier::Medium,
            Tier::Small,
        ]
        .into_iter()
        .find(|&t| t.tables().4 <= ram / 3.0 && (t.tables().0 <= 7 || present(t)))
        .unwrap_or(Tier::Tiny)
    }
}

#[derive(Args, Clone)]
#[command(next_help_heading = "Tables")]
struct TableArgs {
    /// Directory of the pruning tables (default: $EGGG_TABLES, else ./tables). Missing tables
    /// are generated and saved there (once; the large ones take a few minutes).
    #[arg(long, value_name = "DIR")]
    tables: Option<PathBuf>,
    /// Table set by memory use.
    #[arg(long, value_enum, default_value_t = Tier::Auto)]
    tier: Tier,
    /// Override the tier's phase-3 H48 table: edge-orientation bits (0-11; 30 MB * 2^K).
    #[arg(long, value_name = "K")]
    h48: Option<u32>,
    /// Use the joint phase-2 table (tau x U/D centres x PLL parity, 1.05 GB) whatever the
    /// tier: far fewer phase-2 nodes.
    #[arg(long)]
    p2_joint: bool,
    /// Hold the joint phase-2 table's distances exactly (2.1 GB instead of 1.05 GB; made once
    /// from the mod-3 table, saved as p2-joint-exact.tab): phase-1 endpoints become phase-2
    /// candidates with one table read per bound instead of a walk down the table. Default:
    /// the tier's (true in huge and larger).
    #[arg(long, value_name = "BOOL")]
    p2_exact: Option<bool>,
    /// Use the large phase-1 staging table (1.2 GB, no symmetry reduction; faster lookups)
    /// whatever the tier.
    #[arg(long)]
    p1_big: bool,
    /// Phase-1 pattern table (searches per colour pair, pruned by the distance to staged
    /// centres with the pair in a pattern phase 2 can solve): none, the symmetry-reduced one
    /// (0.9 GB) or the large one (41 GB).
    #[arg(long, value_enum)]
    p1_pat: Option<PatArg>,
    /// Phase-1 move-mask table (per staging state, the moves that decrease and that keep its
    /// distance: a node reads one entry instead of ~20 child entries): none, the
    /// symmetry-reduced one (1.6 GB; the default in the large and larger tiers) or the large
    /// one (18.9 GB, the decreasing moves only; implies --p1-big). With --p1-pat: the pair
    /// table instead (pattern-distance-decreasing moves, 17.5 GB).
    #[arg(long, value_enum)]
    p1_mask: Option<PatArg>,
    /// Phase-1 wing-path lookahead (with the move masks): per centre state, the wing slot
    /// sets its staging paths need, up to this distance (0: none; 6: 0.42 GB, 7: 1.8 GB,
    /// built in a few seconds; 8: 15.2 GB, built once in about a minute (peak 15.0 GB)
    /// and saved as p1-wingb-k8s6.tab; default: the tier's).
    #[arg(long, value_name = "K")]
    p1_wing: Option<u8>,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, ValueEnum)]
enum PatArg {
    None,
    Small,
    Big,
}

impl TableArgs {
    /// (h48 K, joint phase-2 table, phase-1 move masks, large phase-1 staging table,
    /// wing-path distance) from the tier and the overrides.
    fn config(&self) -> (u32, JointTable, bool, bool, u8) {
        let (k, joint, masks, wing, _) = self.tier.resolve(&self.dir()).tables();
        let joint = match (joint, self.p2_exact) {
            (_, Some(true)) | (JointTable::Exact, None) => JointTable::Exact,
            (JointTable::None, _) if !self.p2_joint => JointTable::None,
            _ => JointTable::Mod3,
        };
        (
            self.h48.unwrap_or(k),
            joint,
            masks,
            self.p1_big,
            self.p1_wing.unwrap_or(wing),
        )
    }

    fn load(&self, threads: usize, verbose: bool) -> Tables {
        let (k, joint, masks, big, wing_k) = self.config();
        if verbose {
            eprintln!(
                "tables: h48 h{k}{}{}{}{} ({:?} tier)",
                match joint {
                    JointTable::None => "",
                    JointTable::Mod3 => ", joint phase 2",
                    JointTable::Exact => ", exact joint phase 2",
                },
                if masks { ", phase-1 move masks" } else { "" },
                if masks && wing_k > 0 {
                    format!(", wing paths <= {wing_k}")
                } else {
                    String::new()
                },
                if big { ", large phase 1" } else { "" },
                self.tier.resolve(&self.dir())
            );
        }
        let pat = match self.p1_pat {
            None | Some(PatArg::None) => None,
            Some(PatArg::Small) => Some(false),
            Some(PatArg::Big) => Some(true),
        };
        if verbose && let Some(b) = pat {
            eprintln!(
                "phase-1 pattern table: {}",
                if b {
                    "large (41 GB)"
                } else {
                    "reduced (0.9 GB)"
                }
            );
        }
        let mask = match self.p1_mask {
            Some(PatArg::None) => None,
            Some(PatArg::Small) => Some(false),
            Some(PatArg::Big) => Some(true),
            // The tiers with masks use the reduced move-mask table (1.6 GB; phase 1 ~10x
            // faster with the same endpoints), unless a pattern table is asked for (that
            // combination selects the 17.5 GB pair-mask table).
            None if pat.is_none() && masks => Some(false),
            None => None,
        };
        if verbose && let Some(b) = mask {
            eprintln!(
                "phase-1 move-mask table: {}",
                if b {
                    "large (18.9 GB)"
                } else {
                    "reduced (1.6 GB)"
                }
            );
        }
        // (The reduced masks need only the symmetry-reduced staging table, 25 MB; the large
        // ones the large one.)
        load_tables_with(
            &self.dir(),
            k,
            joint,
            big || (pat.is_none() && mask == Some(true)),
            P1Extra { pat, mask, wing_k },
            threads,
            verbose,
        )
    }

    fn dir(&self) -> PathBuf {
        self.tables
            .clone()
            .or_else(|| std::env::var_os("EGGG_TABLES").map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("tables"))
    }
}

#[derive(Args, Clone)]
#[command(next_help_heading = "Search")]
struct SearchArgs {
    /// Worker threads (default: all cores).
    #[arg(short = 'j', long)]
    threads: Option<usize>,
    /// Phase-1 endpoints collected in the first round (doubled each round).
    #[arg(long, default_value_t = Options::default().endpoints)]
    endpoints: usize,
    /// Reductions collected in the first round (doubled each round).
    #[arg(long, default_value_t = Options::default().reductions)]
    reductions: usize,
    /// Search each new phase-1 length's jobs in move order, not best first (for comparisons).
    #[arg(long, hide = true)]
    no_job_order: bool,
    /// Also print every solution as long as the best so far (one per reduction; identical
    /// move sequences once), not only shorter ones. Phase 3 then also searches the best
    /// length, so improvements can come a little later.
    #[arg(long)]
    ties: bool,
    /// Do not use the position's own symmetries (by default, a symmetric position's phase 1
    /// searches one move prefix per orbit of equivalent ones; for comparisons).
    #[arg(long)]
    no_symmetry: bool,
}

#[derive(Args, Clone)]
#[command(next_help_heading = "Output")]
struct OutputArgs {
    /// Move notation of the output.
    #[arg(short = 'n', long, value_enum, default_value_t = Style::Wca)]
    notation: Style,
    /// Print only the solution.
    #[arg(short, long, conflicts_with = "verbose")]
    quiet: bool,
    /// Print search statistics.
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Args)]
struct SolveArgs {
    #[command(flatten)]
    input: InputArgs,
    #[command(flatten)]
    limits: LimitArgs,
    #[command(flatten)]
    search: SearchArgs,
    #[command(flatten)]
    tables: TableArgs,
    #[command(flatten)]
    output: OutputArgs,
}

#[derive(Args)]
struct GenArgs {
    #[command(flatten)]
    tables: TableArgs,
    /// Worker threads (default: all cores).
    #[arg(short = 'j', long)]
    threads: Option<usize>,
}

#[derive(Args)]
struct BenchArgs {
    /// Number of random positions.
    #[arg(short = 'N', long, default_value_t = 10)]
    count: usize,
    #[arg(long, default_value_t = 2026)]
    seed: u64,
    #[command(flatten)]
    limits: LimitArgs,
    #[command(flatten)]
    search: SearchArgs,
    #[command(flatten)]
    tables: TableArgs,
    /// Write all measurements as JSON to this file.
    #[arg(long, value_name = "FILE")]
    json: Option<PathBuf>,
    /// Print each position's result.
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Args)]
struct ShowArgs {
    #[command(flatten)]
    input: InputArgs,
}

struct Position {
    label: String,
    facelets: Facelets,
}

fn read_positions(input: &InputArgs) -> Result<Vec<Position>, String> {
    if let Some(n) = input.random {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(input.seed);
        return Ok((0..n)
            .map(|i| {
                let f = Cube::random(&mut rng).to_facelets();
                Position {
                    label: format!("random #{} (seed {})", i + 1, input.seed),
                    facelets: f,
                }
            })
            .collect());
    }
    let lines: Vec<String> = if input.position.is_empty() {
        std::io::stdin()
            .lock()
            .lines()
            .map_while(Result::ok)
            .map(|l| l.split('#').next().unwrap_or("").trim().to_string())
            .filter(|l| !l.is_empty())
            .collect()
    } else {
        input.position.clone()
    };
    lines
        .into_iter()
        .map(|l| {
            parse_position(&l).map(|f| Position {
                label: l,
                facelets: f,
            })
        })
        .collect()
}

fn parse_position(s: &str) -> Result<Facelets, String> {
    let compact: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    if compact.len() == 96 && !s.contains(' ') {
        return Facelets::parse(&compact);
    }
    let turns = parse_moves(s)?;
    Ok(turns
        .iter()
        .fold(Cube::SOLVED.to_facelets(), |f, &t| f.apply_turn(t)))
}

fn format_solution(s: &Solution, style: Style) -> String {
    let moves: Vec<_> = s
        .moves
        .iter()
        .map(|&m| eggg_solver::moves::move_of(m as usize))
        .collect();
    format_moves(&moves, style)
}

fn setup_threads(j: Option<usize>) -> usize {
    if let Some(j) = j {
        eggg_solver::par::set_threads(j);
    }
    eggg_solver::par::threads()
}

/// No time limit: about a century.
const FOREVER: Duration = Duration::from_secs(100 * 365 * 86_400);

/// The default time limit of `bench`, in seconds.
const BENCH_TIME: f64 = 5.0;

fn options(limits: &LimitArgs, search: &SearchArgs, threads: usize) -> Options {
    Options {
        time_limit: limits.time.map_or(FOREVER, Duration::from_secs_f64),
        target_length: limits.max_length,
        hard_limit: limits.hard_limit.map(Duration::from_secs_f64),
        threads,
        endpoints: search.endpoints,
        reductions: search.reductions,
        order_jobs: !search.no_job_order,
        ties: search.ties,
        pool_budget: pool_budget(),
        symmetry: !search.no_symmetry,
    }
}

/// Memory for the solver's candidate and reduction pools, once the tables are loaded: half of
/// the memory available beyond 8 GB of headroom, at least 1 GB. Available: Linux's
/// MemAvailable, else a quarter of the physical memory; `EGGG_PHYS_GB=N` sizes it as on a
/// machine with N GB (N GB less this process and 4 GB of other use).
fn pool_budget() -> usize {
    const GB: f64 = (1u64 << 30) as f64;
    // SAFETY: sysconf has no preconditions.
    let ram = unsafe {
        libc::sysconf(libc::_SC_PHYS_PAGES) as f64 * libc::sysconf(libc::_SC_PAGESIZE) as f64
    };
    let kb = |key: &str, text: &str| {
        text.lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<f64>().ok())
            .map(|v| v * 1024.0)
    };
    let avail = match std::env::var("EGGG_PHYS_GB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        Some(n) => {
            let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
            n * GB - kb("VmRSS:", &status).unwrap_or(0.0) - 4.0 * GB
        }
        None => std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|m| kb("MemAvailable:", &m))
            .unwrap_or(ram / 4.0),
    };
    ((avail - 8.0 * GB) / 2.0).max(GB) as usize
}

fn solve_one(
    tables: &Tables,
    opts: &Options,
    f: &Facelets,
    mut on: impl FnMut(&Solution),
) -> (Option<Solution>, eggg_solver::solver::Stats) {
    let (cube, _) = match f.to_cube() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("invalid position: {e}");
            return (None, Default::default());
        }
    };
    let mut solver = Solver::new(tables, opts.clone());
    let sol = solver.solve(&cube, &mut |s| on(s));
    if let Some(s) = &sol {
        assert!(
            verify(f, &s.moves),
            "internal error: the solution does not solve the cube"
        );
    }
    (sol, solver.stats)
}

fn cmd_solve(a: SolveArgs) -> Result<(), String> {
    let threads = setup_threads(a.search.threads);
    let positions = read_positions(&a.input)?;
    let tables = a.tables.load(threads, !a.output.quiet);
    let opts = options(&a.limits, &a.search, threads);
    let style = a.output.notation;
    for p in positions {
        if !a.output.quiet && positions_label_useful(&p.label) {
            println!("# {}", p.label);
        }
        let (quiet, ties) = (a.output.quiet, opts.ties);
        let (sol, stats) = solve_one(&tables, &opts, &p.facelets, |s| {
            if !quiet {
                println!(
                    "{:9.3} s  {:2} moves  phases {:?}  {}",
                    s.elapsed.as_secs_f64(),
                    s.len(),
                    s.phases,
                    format_solution(s, style)
                );
            } else if ties {
                println!("{}", format_solution(s, style));
            }
        });
        match sol {
            Some(s) => {
                println!("{}", format_solution(&s, style));
                if !a.output.quiet {
                    println!(
                        "{} moves; found at {:.3} s",
                        s.len(),
                        s.elapsed.as_secs_f64()
                    );
                }
            }
            None => println!("no solution found"),
        }
        if a.output.verbose {
            eprintln!("{stats:?}");
            eprintln!(
                "symmetry: stabiliser of order {} ({} rotations), pruned by {} ({} of {} phase-1 job prefixes skipped)",
                stats.sym_order, stats.sym_rotations, stats.sym_used, stats.jobs_pruned, stats.jobs
            );
        }
        std::io::stdout().flush().ok();
    }
    Ok(())
}

fn positions_label_useful(l: &str) -> bool {
    l.len() < 400
}

fn cmd_gen(a: GenArgs) -> Result<(), String> {
    let threads = setup_threads(a.threads);
    let t = Instant::now();
    let _ = a.tables.load(threads, true);
    eprintln!("all tables present ({:.1} s)", t.elapsed().as_secs_f64());
    Ok(())
}

fn cmd_bench(mut a: BenchArgs) -> Result<(), String> {
    let threads = setup_threads(a.search.threads);
    let tables = a.tables.load(threads, true);
    let time = *a.limits.time.get_or_insert(BENCH_TIME);
    let mut opts = options(&a.limits, &a.search, threads);
    opts.hard_limit
        .get_or_insert(Duration::from_secs_f64(time * 4.0));
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(a.seed);
    let checkpoints = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0];
    let checkpoints: Vec<f64> = checkpoints
        .into_iter()
        .filter(|&c| c <= time + 1e-9)
        .collect();
    let mut rows = vec![];
    for i in 0..a.count {
        let f = Cube::random(&mut rng).to_facelets();
        let mut history: Vec<(f64, usize)> = vec![];
        let t0 = Instant::now();
        let (sol, stats) = solve_one(&tables, &opts, &f, |s| {
            history.push((s.elapsed.as_secs_f64(), s.len()))
        });
        let wall = t0.elapsed().as_secs_f64();
        let final_len = sol.as_ref().map(|s| s.len()).unwrap_or(0);
        if a.verbose {
            eprintln!(
                "#{:<3} {:2} moves (first {:2} at {:.3} s) phases {:?} {:.2} s  nodes {}/{}/{}",
                i + 1,
                final_len,
                history.first().map(|h| h.1).unwrap_or(0),
                history.first().map(|h| h.0).unwrap_or(0.0),
                sol.as_ref().map(|s| s.phases).unwrap_or([0; 3]),
                wall,
                stats.nodes1,
                stats.nodes2,
                stats.nodes3
            );
        }
        rows.push((history, final_len, stats, wall));
    }
    let n = rows.len() as f64;
    println!(
        "positions: {}  threads: {}  time limit: {} s",
        rows.len(),
        threads,
        time
    );
    let at = |h: &Vec<(f64, usize)>, t: f64| h.iter().filter(|x| x.0 <= t).map(|x| x.1).min();
    for &c in &checkpoints {
        let v: Vec<usize> = rows.iter().filter_map(|r| at(&r.0, c)).collect();
        if v.len() == rows.len() {
            println!(
                "  mean length at {:5.2} s: {:.3}",
                c,
                v.iter().sum::<usize>() as f64 / n
            );
        } else {
            println!(
                "  mean length at {:5.2} s: - ({} of {} solved)",
                c,
                v.len(),
                rows.len()
            );
        }
    }
    let firsts: Vec<(f64, usize)> = rows.iter().filter_map(|r| r.0.first().copied()).collect();
    let nf = firsts.len().max(1) as f64;
    println!(
        "  first solution: {:.3} moves at {:.3} s",
        firsts.iter().map(|x| x.1 as f64).sum::<f64>() / nf,
        firsts.iter().map(|x| x.0).sum::<f64>() / nf
    );
    let solved: Vec<f64> = rows
        .iter()
        .filter(|r| r.1 > 0)
        .map(|r| r.1 as f64)
        .collect();
    let fin = solved.iter().sum::<f64>() / solved.len().max(1) as f64;
    println!(
        "  final: {:.3} moves ({} of {} solved)",
        fin,
        solved.len(),
        rows.len()
    );
    let (n1, n2, n3) = rows.iter().fold((0u64, 0u64, 0u64), |a, r| {
        (a.0 + r.2.nodes1, a.1 + r.2.nodes2, a.2 + r.2.nodes3)
    });
    let wall: f64 = rows.iter().map(|r| r.3).sum();
    let tp: Vec<f64> = (0..3)
        .map(|i| rows.iter().map(|r| r.2.time[i].as_secs_f64()).sum::<f64>())
        .collect();
    let tt: f64 = tp.iter().sum::<f64>().max(1e-9);
    println!(
        "  time split: phase 1 {:.0}%, phase 2 {:.0}%, phase 3 {:.0}%",
        100.0 * tp[0] / tt,
        100.0 * tp[1] / tt,
        100.0 * tp[2] / tt
    );
    println!(
        "  nodes/position: p1 {:.3e} p2 {:.3e} p3 {:.3e}  ({:.3e} nodes/s)",
        n1 as f64 / n,
        n2 as f64 / n,
        n3 as f64 / n,
        (n1 + n2 + n3) as f64 / wall
    );
    if let Some(path) = a.json {
        let mut s = String::from("{\"positions\":[");
        for (i, r) in rows.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            let hist: Vec<String> = r.0.iter().map(|(t, l)| format!("[{t:.4},{l}]")).collect();
            s.push_str(&format!(
                "{{\"final\":{},\"wall\":{:.4},\"nodes\":[{},{},{}],\"time\":[{:.4},{:.4},{:.4}],\"history\":[{}]}}",
                r.1,
                r.3,
                r.2.nodes1,
                r.2.nodes2,
                r.2.nodes3,
                r.2.time[0].as_secs_f64(),
                r.2.time[1].as_secs_f64(),
                r.2.time[2].as_secs_f64(),
                hist.join(",")
            ));
        }
        s.push_str(&format!(
            "],\"threads\":{},\"time\":{},\"seed\":{}}}",
            threads, time, a.seed
        ));
        std::fs::write(path, s).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn cmd_bench3(a: Bench3Args) -> Result<(), String> {
    use eggg_solver::cube3::Cube3;
    use eggg_solver::phase3::Search3;
    use rand::Rng;
    let threads = setup_threads(a.threads);
    let tables = a.tables.load(threads, true);
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(a.seed);
    let cubes: Vec<Cube3> = (0..a.count)
        .map(|_| {
            let mut x = Cube3::SOLVED;
            for _ in 0..60 {
                x = x.apply(rng.gen_range(0..18));
            }
            x
        })
        .collect();
    let results = std::sync::Mutex::new(vec![]);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= cubes.len() {
                        break;
                    }
                    let t = Instant::now();
                    let mut srch = Search3::new(&tables.p3);
                    let h = srch.lower_bound(&cubes[i]);
                    let sol = srch.solve(&cubes[i], 20).expect("no solution");
                    assert_eq!(cubes[i].apply_moves(&sol), Cube3::SOLVED);
                    results.lock().unwrap().push((
                        i,
                        h,
                        sol.len(),
                        srch.nodes,
                        t.elapsed().as_secs_f64(),
                    ));
                }
            });
        }
    });
    let mut r = results.into_inner().unwrap();
    r.sort_by_key(|x| x.0);
    for x in &r {
        eprintln!(
            "#{:<3} h {:2} optimal {:2} nodes {:12} {:8.3} s",
            x.0, x.1, x.2, x.3, x.4
        );
    }
    let n = r.len() as f64;
    let nodes: u64 = r.iter().map(|x| x.3).sum();
    let cpu: f64 = r.iter().map(|x| x.4).sum();
    println!(
        "h48 h{}: {} cubes, mean length {:.2}, mean root bound {:.2}, mean nodes {:.3e}, {:.3} s/cube (thread time), {:.3e} nodes/s/thread, wall {:.2} s",
        a.tables.config().0,
        r.len(),
        r.iter().map(|x| x.2 as f64).sum::<f64>() / n,
        r.iter().map(|x| x.1 as f64).sum::<f64>() / n,
        nodes as f64 / n,
        cpu / n,
        nodes as f64 / cpu,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn cmd_show(a: ShowArgs) -> Result<(), String> {
    for p in read_positions(&a.input)? {
        println!("{}", p.facelets.to_string_letters());
    }
    Ok(())
}

fn main() {
    // Exit quietly when the output pipe closes.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    let cli = Cli::parse();
    let r = match cli.cmd {
        Cmd::Solve(a) => cmd_solve(a),
        Cmd::Gen(a) => cmd_gen(a),
        Cmd::Bench(a) => cmd_bench(a),
        Cmd::Show(a) => cmd_show(a),
        Cmd::Bench3(a) => cmd_bench3(a),
    };
    if let Err(e) = r {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
