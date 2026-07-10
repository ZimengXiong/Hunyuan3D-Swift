import Foundation

// hy3d — one executable, five subcommands. Hand-rolled dispatch (no ArgumentParser dependency).
setbuf(stdout, nil)   // unbuffered so progress prints appear promptly

let argv = Array(CommandLine.arguments.dropFirst())
guard let sub = argv.first else { printUsage(); exit(2) }
let rest = Array(argv.dropFirst())

do {
    switch sub {
    case "shape":         try cmdShape(Args(rest))
    case "paint":         try cmdPaint(Args(rest, bools: ["no-superres"]))
    case "generate":      try cmdGenerate(Args(rest, bools: ["no-superres"]))
    case "parity-shape":  try cmdParityShape(Args(rest))
    case "parity-paint":  try cmdParityPaint(Args(rest))
    case "-h", "--help", "help":
        printUsage()
    default:
        FileHandle.standardError.write("unknown subcommand: \(sub)\n".data(using: .utf8)!)
        printUsage()
        exit(2)
    }
} catch let e as CLIError {
    die(e.message)
} catch {
    die("\(error)")
}
