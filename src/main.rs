use clap::{Args, Parser, Subcommand};
use eggg_solver::cube::{Cube, Facelets};
use eggg_solver::notation::{Style, format_moves, parse_moves};
use eggg_solver::solver::{Options, Solution, Solver, Tables, load_tables, verify};
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
    /// Search for at most this many seconds per position.
    #[arg(short = 't', long, value_name = "SECONDS", default_value_t = 5.0)]
    time: f64,
    /// Stop as soon as a solution of at most this many moves is found.
    #[arg(short = 'l', long, value_name = "MOVES")]
    max_length: Option<usize>,
}

#[derive(Args, Clone)]
#[command(next_help_heading = "Tables")]
struct TableArgs {
    /// Directory of the pruning tables (default: $EGGG_TABLES, else ./tables). Missing tables
    /// are generated and saved there.
    #[arg(long, value_name = "DIR")]
    tables: Option<PathBuf>,
    /// Phase-3 H48 table: number of edge-orientation bits (0-11; table size 30 MB * 2^K).
    #[arg(long, value_name = "K", default_value_t = 2)]
    h48: u32,
}

impl TableArgs {
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
    /// Phase-1 endpoints collected per round.
    #[arg(long, default_value_t = Options::default().endpoints)]
    endpoints: usize,
    /// Reductions collected per round.
    #[arg(long, default_value_t = Options::default().reductions)]
    reductions: usize,
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

fn options(limits: &LimitArgs, search: &SearchArgs, threads: usize) -> Options {
    Options {
        time_limit: Duration::from_secs_f64(limits.time),
        target_length: limits.max_length,
        threads,
        endpoints: search.endpoints,
        reductions: search.reductions,
    }
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
    let tables = load_tables(&a.tables.dir(), a.tables.h48, threads, !a.output.quiet);
    let opts = options(&a.limits, &a.search, threads);
    let style = a.output.notation;
    for p in positions {
        if !a.output.quiet && positions_label_useful(&p.label) {
            println!("# {}", p.label);
        }
        let quiet = a.output.quiet;
        let (sol, stats) = solve_one(&tables, &opts, &p.facelets, |s| {
            if !quiet {
                println!(
                    "{:9.3} s  {:2} moves  phases {:?}  {}",
                    s.elapsed.as_secs_f64(),
                    s.len(),
                    s.phases,
                    format_solution(s, style)
                );
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
    let _ = load_tables(&a.tables.dir(), a.tables.h48, threads, true);
    eprintln!("all tables present ({:.1} s)", t.elapsed().as_secs_f64());
    Ok(())
}

fn cmd_bench(a: BenchArgs) -> Result<(), String> {
    let threads = setup_threads(a.search.threads);
    let tables = load_tables(&a.tables.dir(), a.tables.h48, threads, true);
    let opts = options(&a.limits, &a.search, threads);
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(a.seed);
    let checkpoints = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0];
    let checkpoints: Vec<f64> = checkpoints
        .into_iter()
        .filter(|&c| c <= a.limits.time + 1e-9)
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
        a.limits.time
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
    let first_t: f64 = rows
        .iter()
        .map(|r| r.0.first().map(|x| x.0).unwrap_or(f64::NAN))
        .sum::<f64>()
        / n;
    let first_l: f64 = rows
        .iter()
        .map(|r| r.0.first().map(|x| x.1 as f64).unwrap_or(f64::NAN))
        .sum::<f64>()
        / n;
    let fin: f64 = rows.iter().map(|r| r.1 as f64).sum::<f64>() / n;
    println!("  first solution: {:.3} moves at {:.3} s", first_l, first_t);
    println!("  final: {:.3} moves", fin);
    let (n1, n2, n3) = rows.iter().fold((0u64, 0u64, 0u64), |a, r| {
        (a.0 + r.2.nodes1, a.1 + r.2.nodes2, a.2 + r.2.nodes3)
    });
    let wall: f64 = rows.iter().map(|r| r.3).sum();
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
                "{{\"final\":{},\"wall\":{:.4},\"nodes\":[{},{},{}],\"history\":[{}]}}",
                r.1,
                r.3,
                r.2.nodes1,
                r.2.nodes2,
                r.2.nodes3,
                hist.join(",")
            ));
        }
        s.push_str(&format!(
            "],\"threads\":{},\"time\":{},\"seed\":{}}}",
            threads, a.limits.time, a.seed
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
    let tables = load_tables(&a.tables.dir(), a.tables.h48, threads, true);
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
        a.tables.h48,
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
