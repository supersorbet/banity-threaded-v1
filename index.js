#! /usr/bin/env node

import VanityEth from "./libs/VanityEth.js";
import ora from "ora";
import cluster from "cluster";
import TimeFormat from "hh-mm-ss";
import { cpus } from "os";
import Yargs from "yargs";
import process from "process";
import fs from "fs";

const numCPUs = cpus().length > 1 ? cpus().length - 1 : 1;
let threadCount, args, spinner, logStream, walletsFound = 0, addps = 0;

// Move processFoundWallet to global scope
async function processFoundWallet(wallet, vanityContract, isChecksum, maxNonce) {
  if (!wallet || !vanityContract) return wallet;
  
  try {
    const { nonce, contractAddress } = await VanityEth.findContractNonce(
      wallet.address,
      vanityContract,
      isChecksum,
      maxNonce,
      () => { if (cluster.isMaster) addps++; }
    );
    wallet.contractNonce = nonce;
    wallet.vanityContract = contractAddress;
    return wallet;
  } catch (error) {
    console.error(`Could not find matching contract address: ${error.message}`);
    return null;
  }
}

const argv = Yargs(process.argv.slice(2))
  .usage("Usage: $0 [mode] [options]")
  .example("$0 -cpu 6 0000", "use 6 CPU threads for pattern ABC")
  .example("$0 -cuda 4 0000", "use NVIDIA GPU acceleration")
  .example("$0 -opencl 4 0000", "use AMD/Intel GPU acceleration")
  .example("$0 -cpu 6 0000 --vanity-contract DEAD", "find EOA with pattern 0000 that can deploy contract with pattern DEAD")
  .command("* [pattern]", "pattern to search for", (yargs) => {
    yargs.positional("pattern", {
      describe: "vanity pattern to search for",
      type: "string",
    });
  })
  .number("cuda")
  .describe("cuda", "use NVIDIA GPU acceleration")
  .number("opencl")
  .describe("opencl", "use AMD/Intel GPU acceleration")
  .number("cpu")
  .describe("cpu", "use CPU with specified number of threads")
  .string("vanity-contract")
  .describe("vanity-contract", "find a vanity contract address deployable from the vanity EOA")
  .number("max-nonce")
  .default("max-nonce", 1000000)
  .describe("max-nonce", "maximum nonce to try for contract address")
  .alias("n", "count")
  .number("n")
  .describe("n", "number of wallets to generate")
  .alias("x", "contract")
  .boolean("contract")
  .describe("contract", "generate contract address")
  .alias("s", "checksum")
  .boolean("checksum")
  .describe("checksum", "use checksum address")
  .alias("l", "log")
  .boolean("l")
  .describe("l", "log output to file")
  .help("h")
  .alias("h", "help")
  .check((argv) => {
    const modes = ['cpu', 'cuda', 'opencl'].filter(mode => argv[mode] !== undefined);
    if (modes.length > 1) {
      throw new Error("Cannot use multiple computation modes at the same time. Choose one of: -cpu, -cuda, or -opencl");
    }
    if (!argv.pattern && !argv._.length) {
      throw new Error("Pattern is required");
    }
    return true;
  }).argv;

if (cluster.isMaster) {
  args = {
    input: argv.pattern || argv._[0],
    isChecksum: argv.checksum ? true : false,
    numWallets: argv.count ? argv.count : 1,
    isContract: argv.contract ? true : false,
    log: argv.log ? true : false,
    logFname: argv.log ? "VanityEth-log-" + Date.now() + ".txt" : "",
    useCUDA: argv.cuda !== undefined,
    useOpenCL: argv.opencl !== undefined,
    vanityContract: argv["vanity-contract"],
    maxNonce: argv["max-nonce"]
  };
  
  threadCount = argv.cpu || argv.cuda || argv.opencl || numCPUs;
  
  if (!VanityEth.isValidHex(args.input)) {
    console.error(args.input + " is not valid hexadecimal");
    process.exit(1);
  }

  if (args.vanityContract && !VanityEth.isValidHex(args.vanityContract)) {
    console.error(args.vanityContract + " is not valid hexadecimal");
    process.exit(1);
  }

  if (args.log) {
    console.log("logging into " + args.logFname);
    logStream = fs.createWriteStream(args.logFname, { flags: "a" });
  }

  spinner = ora("generating vanity address 1/" + args.numWallets).start();
  
  setInterval(function () {
    spinner.text =
      "Approximate ETA for an account " +
      TimeFormat.fromS(
        Math.pow(16, 20) / Math.pow(16, 20 - args.input.length) / addps,
        "hh:mm:ss"
      );
    addps = 0;
  }, 1000);

  if (args.useOpenCL) {
    // OpenCL GPU mode (AMD/Intel)
    import("./libs/VanityEthOpenCL.js").then((VanityEthOpenCL) => {
      const runOpenCL = async () => {
        try {
          for (let i = 0; i < args.numWallets; i++) {
            const account = await VanityEthOpenCL.default.getVanityWalletOpenCL(
              args.input,
              args.isChecksum,
              args.isContract,
              threadCount,
              () => { addps++; }
            );
            
            const processedAccount = await processFoundWallet(account, args.vanityContract, args.isChecksum, args.maxNonce);
            if (processedAccount) {
              spinner.succeed(JSON.stringify({ account: processedAccount }));
              if (args.log) logStream.write(JSON.stringify({ account: processedAccount }) + "\n");
              walletsFound++;
            }
            
            if (walletsFound >= args.numWallets) {
              cleanup();
            } else {
              spinner.text = "generating vanity address " + (walletsFound + 1) + "/" + args.numWallets;
              spinner.start();
            }
          }
        } catch (error) {
          console.error("OpenCL error:", error);
          console.log("Falling back to CPU mode...");
          startCPUWorkers();
        }
      };
      runOpenCL();
    }).catch((error) => {
      console.error("OpenCL mode not available:", error.message);
      console.log("Falling back to CPU mode...");
      startCPUWorkers();
    });
  } else if (args.useCUDA) {
    // CUDA GPU mode
    import("./libs/VanityEthCUDA.js").then((VanityEthCUDA) => {
      const runCUDA = async () => {
        try {
          for (let i = 0; i < args.numWallets; i++) {
            const account = await VanityEthCUDA.default.getVanityWalletCUDA(
              args.input,
              args.isChecksum,
              args.isContract,
              threadCount,
              () => { addps++; }
            );
            
            const processedAccount = await processFoundWallet(account, args.vanityContract, args.isChecksum, args.maxNonce);
            if (processedAccount) {
              spinner.succeed(JSON.stringify({ account: processedAccount }));
              if (args.log) logStream.write(JSON.stringify({ account: processedAccount }) + "\n");
              walletsFound++;
            }
            
            if (walletsFound >= args.numWallets) {
              cleanup();
            } else {
              spinner.text = "generating vanity address " + (walletsFound + 1) + "/" + args.numWallets;
              spinner.start();
            }
          }
        } catch (error) {
          console.error("CUDA error:", error);
          console.log("Falling back to CPU mode...");
          startCPUWorkers();
        }
      };
      runCUDA();
    }).catch((error) => {
      console.error("CUDA mode not available:", error.message);
      console.log("Falling back to CPU mode...");
      startCPUWorkers();
    });
  } else {
    // CPU mode - multiple processes
    startCPUWorkers();
  }
} else {
  // Worker process
  const worker_env = process.env;
  const runWorker = async () => {
    while (true) {
      if (process.send) {
        const account = await VanityEth.getVanityWallet(
          worker_env.input,
          worker_env.isChecksum === "true",
          worker_env.isContract === "true",
          function () {
            process.send({
              counter: true,
            });
          }
        );

        // Process the wallet in the worker
        const processedAccount = await processFoundWallet(
          account,
          worker_env.vanityContract,
          worker_env.isChecksum === "true",
          parseInt(worker_env.maxNonce || "1000000")
        );

        if (processedAccount) {
          process.send({ account: processedAccount });
        }
      }
    }
  };
  runWorker().catch(console.error);
}

function startCPUWorkers() {
  for (let i = 0; i < threadCount; i++) {
    const worker_env = {
      input: args.input,
      isChecksum: args.isChecksum,
      isContract: args.isContract,
      vanityContract: args.vanityContract,
      maxNonce: args.maxNonce
    };
    const proc = cluster.fork(worker_env);
    proc.on("message", function (message) {
      if (message.account) {
        spinner.succeed(JSON.stringify(message));
        if (args.log) logStream.write(JSON.stringify(message) + "\n");
        walletsFound++;
        if (walletsFound >= args.numWallets) {
          cleanup();
        }
        spinner.text =
          "generating vanity address " +
          (walletsFound + 1) +
          "/" +
          args.numWallets;
        spinner.start();
      } else if (message.counter) {
        addps++;
      }
    });
  }
}

process.stdin.resume();
const cleanup = function (options, err) {
  if (err) console.log(err.stack);
  for (const id in cluster.workers) cluster.workers[id].process.kill();
  process.exit();
};
process.on("exit", cleanup.bind(null, {}));
process.on("SIGINT", cleanup.bind(null, {}));
process.on("uncaughtException", cleanup.bind(null, {}));

