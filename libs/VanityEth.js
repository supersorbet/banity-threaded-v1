import crypto from "crypto";
import ethUtils from "ethereumjs-util";

// Add sleep utility function
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

var ERRORS = {
  invalidHex: "Invalid hex input",
};
var getRandomWallet = function () {
  var randbytes = crypto.randomBytes(32);
  var address = "0x" + ethUtils.privateToAddress(randbytes).toString("hex");
  return { address: address, privKey: randbytes.toString("hex") };
};
var isValidHex = function (hex) {
  if (!hex.length) return true;
  hex = hex.toUpperCase();
  var re = /^[0-9A-F]+$/g;
  return re.test(hex);
};
var getDeterministicContractAddress = function (address, nonce) {
  return (
    "0x" +
    ethUtils
      .keccak256(ethUtils.rlp.encode([address, nonce]))
      .slice(12)
      .toString("hex")
  );
};
var isValidVanityWallet = function (wallet, input, isChecksum, isContract) {
  var _add = wallet.address;
  if (isContract) {
    var _contractAdd = getDeterministicContractAddress(_add, 0);
    _contractAdd = isChecksum
      ? ethUtils.toChecksumAddress(_contractAdd)
      : _contractAdd;
    wallet.contract = _contractAdd;
    return _contractAdd.substr(2, input.length) == input;
  }
  _add = isChecksum ? ethUtils.toChecksumAddress(_add) : _add;
  return _add.substr(2, input.length) == input;
};
async function findContractNonce(
  address,
  input = "",
  isChecksum = false,
  maxNonce = 1000000,
  counter = function () {}
) {
  if (!isValidHex(input)) throw new Error(ERRORS.invalidHex);
  input = isChecksum ? input : input.toLowerCase();

  for (let nonce = 0; nonce < maxNonce; nonce++) {
    counter();
    let contractAddress = getDeterministicContractAddress(address, nonce);
    if (isChecksum) {
      contractAddress = ethUtils.toChecksumAddress(contractAddress);
    }
    if (contractAddress.substr(2, input.length).toLowerCase() === input.toLowerCase()) {
      return {
        nonce: nonce,
        contractAddress: contractAddress
      };
    }
    
    // Add throttling to prevent CPU overload
    if (nonce % 1000 === 0) {
      await sleep(1);
    }
  }
  throw new Error(`No matching nonce found within ${maxNonce} attempts`);
}
async function getVanityWallet(
  input = "",
  isChecksum = false,
  isContract = false,
  counter = function () {}
) {
  if (!isValidHex(input)) throw new Error(ERRORS.invalidHex);
  input = isChecksum ? input : input.toLowerCase();
  var _wallet = getRandomWallet();
  let attempts = 0;
  
  while (!isValidVanityWallet(_wallet, input, isChecksum, isContract)) {
    counter();
    _wallet = getRandomWallet(isChecksum);
    
    // Add throttling: Every 1000 attempts, sleep for 1ms to prevent CPU overload
    attempts++;
    if (attempts % 1000 === 0) {
      await sleep(1);
    }
  }
  
  if (isChecksum) _wallet.address = ethUtils.toChecksumAddress(_wallet.address);
  return _wallet;
}
export default { getVanityWallet, isValidHex, findContractNonce, ERRORS };

