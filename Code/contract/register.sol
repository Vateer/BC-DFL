pragma solidity ^0.4.21;
contract Register{
    address public owner = 0x0;
    // bool [] nodes;
    mapping(address => bool) nodes;
    address[] private nodes_name;
    int public epoch=-1;
    string public opti;
    string public target_acc;
    uint public worker=0;
    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }
    event NodeRegistered(address indexed node);
    event Log(string message);
    function isRegister(address node) public returns (bool){
        return nodes[node];
    }
    function deleteNode(address node) public{
        nodes[node]=false;
    }
    
    function registerNode(address node) public {
        require(msg.sender == owner);
        require(node != 0x0); // 确认节点地址不为空
        require(nodes[node] == false); // 确认节点未被注册
        nodes[node] = true;
        nodes_name.push(node);
        worker = worker + 1;
        emit NodeRegistered(node);
    }
    
    function clearContract() public {
        require(msg.sender == owner);
        owner = 0x0;
        epoch=-1;
        target_acc="";
        opti="";
        worker=0;
        for (uint i=0;i<nodes_name.length;i++){
            delete nodes[nodes_name[i]];
        }
        nodes_name.length=0;
    }

    function ownerRegister() public {
        require(owner==0x0);
        owner=msg.sender;
    }
    function setEpoch(int inp) public{
        require(owner==msg.sender);
        epoch = inp;
    }

    function setOpti(string s) public{
        require(owner==msg.sender);
        opti=s;
    }

    function setTargetAcc(string s) public{
        require(owner==msg.sender);
        target_acc=s;
    }

    function getWorkerNumber() public returns (uint){
        return worker;
    }

    function getOwner() public returns (address){
        return owner;
    }
}