pragma solidity ^0.4.21;

contract Register{
    function getOwner() external returns (address);
    function isRegister(address node) external returns (bool);
    function getWorkerNumber() external returns (uint);
}
contract Aggregrate {
    // 关键字“public”让这些变量可以从外部读取
    address public owner;
    string[] public local_parameter;
    string[] public uploaders;
    int public status=0;
    Register public register;
    uint256 public threshold = 10;
    uint[] public samples_number;
    constructor(address obj_address) public {
        register=Register(obj_address);
    }
    function isRegister(address query_add) public returns (bool){
        return register.isRegister(query_add);
    }
    
    function strConcat(string memory _a, string memory _b) public pure returns (string memory){
        bytes memory _ba = bytes(_a);
        bytes memory _bb = bytes(_b);
        string memory ret = new string(_ba.length + _bb.length);
        bytes memory bret = bytes(ret);
        uint k = 0;
        for (uint ii = 0; ii < _ba.length; ii++) 
            bret[k++] = _ba[ii];
        for (uint i = 0; i < _bb.length; i++) 
            bret[k++] = _bb[i];
        return string(ret);
    }

    function uint256ToString(uint256 _i) internal pure returns (string memory) {
      if (_i == 0) {
          return "0";
      }
      uint256 j = _i;
      uint256 length;
      while (j != 0){
          length++;
          j /= 10;
      }
      bytes memory bstr = new bytes(length);
      uint256 k = length;
      while (_i != 0){
          k = k-1;
          uint8 temp = uint8(48 + uint8(_i - _i / 10 * 10));
          bytes1 b1 = bytes1(temp);
          bstr[k] = b1;
          _i /= 10;
      }
      return string(bstr);
   }

    function get_status() public returns (int){
        return status;
    }

    function get_gradient() public returns (string){
        require(msg.sender == register.getOwner());
        if (status != 1) return "null";
        string memory s = "";
        for (uint i=0;i<local_parameter.length;i++){
            s=strConcat(s,local_parameter[i]);
            s=strConcat(s,"|");
            s=strConcat(s,uint256ToString(samples_number[i]));
            s=strConcat(s,"|");
            s=strConcat(s,uploaders[i]);
            s=strConcat(s,"|");
        }
        // status=2;
        status=0;
        uploaders.length=0;
        local_parameter.length=0;
        samples_number.length=0;
        return s;
    }

    // function get_uploader() public returns (address[]){
    //     require(msg.sender == register.getOwner());
    //     if (status != 2) return;
    //     status=0;
    //     address[] memory temp = uploader; //这里一定要memory
    //     uploader.length = 0;
    //     return temp;
    // }

    function send_gradient(string p, uint sample_number, string uploader) public{
        if (local_parameter.length>=threshold){
            status=1;
            return;
        }
        local_parameter.push(p);
        samples_number.push(sample_number);
        uploaders.push(uploader);
        if (local_parameter.length>=threshold)
            status=1;
    }

    function set_threshold(uint inp) public returns (bool){
        require(msg.sender == register.getOwner());
        threshold = inp;
    }

    function init() public{
        require(msg.sender == register.getOwner());
        threshold = 5;
    }

    function debug_gradient() public returns (string){
        string memory s = "";
        for (uint i=0;i<local_parameter.length;i++){
            s=strConcat(s,local_parameter[i]);
            s=strConcat(s,"|");
        }
        return s;
    }
}
