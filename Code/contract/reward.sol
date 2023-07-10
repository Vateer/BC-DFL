pragma solidity ^0.4.21;

contract Register{
    function getOwner() external returns (address);
    function isRegister(address node) external returns (bool);
    function getWorkerNumber() external returns (uint);
}

contract Reward{
    event LogEvent(int T, string uploader,int acc);
    Register public register;
    constructor(address obj_address) public {
        register=Register(obj_address);
    }
    function log_acc(int T, string uploader, int acc){
        require(msg.sender == register.getOwner());
        emit LogEvent(T,uploader,acc);
    }
}
