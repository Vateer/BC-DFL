from web3 import Web3, HTTPProvider
import json
import os, logging, subprocess, time
import IPFS
logging.getLogger().setLevel(logging.INFO)#解决logging.info不输出的问题
class MyTools():
    def __init__(self) -> None:
        pass

    def load_contract_info(input_path, is_dir=None):
        if not os.path.exists(input_path):
            logging.error("file/path not exists!")
            return None
        logging.info("loading name, abi, bytecodes of contracts")
        paths=[]
        names=[]
        if is_dir:
            for file_name in os.listdir(input_path):
                file_path=os.path.join(input_path,file_name)
                if not os.path.isdir(file_path) and file_name.endswith(".abi"):
                    paths.append(file_path)
                    names.append(os.path.splitext(file_name)[0])
        else:
            paths.append(input_path)
            names.append(os.path.splitext(input_path.split('\\')[-1])[0])
        abis=[]
        addresses=[]
        for file in paths:
            with open(file, "r") as f:
                abis.append(f.readline().replace('\n',''))
                addresses.append(f.readline().replace('\n',''))
            logging.info("search {}".format(file))
        return names, abis, addresses
                    




class MyEth:
    # class Register:
    #     def __init__(self, handler) -> None:
    #         self.handler=handler


    #     def register_owner(self, address):
    #         address.functions.registerNode(address)

    def __init__(self, port):
        #连接以太坊
        self.web3=Web3(HTTPProvider("http://localhost:"+str(port)))
        # self.abi_dict={}
        self.cont_dict={}
        self.unlockAcc=""
        # print(self.web3.eth.get_block(0)) 获得块
        # print(self.web3.eth.get_block_number()) 获得长度
        if self.web3.eth.get_block(0) is None:
            logging.error("fail to connect!")
        else:
            logging.info("successfully connect")
    def load_contract(self, name, abi_value, contract_address):
        # self.abi_dict[name]=json.load(abi_value)
        # self.myContractAddr = Web3.toChecksumAddress(contract_address)
        self.cont_dict[name]=self.web3.eth.contract(address=Web3.to_checksum_address(contract_address),abi=abi_value)

    def unlockAccount(self, addr, passw):
        self.web3.geth.personal.unlock_account(Web3.to_checksum_address(addr), passw)
        # self.unlockAcc=addr
        self.unlockAcc=Web3.to_checksum_address(addr)
        # print(addr)
        # print(Web3.to_checksum_address(self.unlockAcc))
        # print(self.unlockAcc)
        # print(type(Web3.to_checksum_address(addr)))
        

    def get(self, name):
        return self.cont_dict[name].functions.get().call({'from':self.unlockAcc})
    
    def set(self, name, value):
        # tx_hash = self.cont_dict[name].functions.set(value).transact({'from':Web3.to_checksum_address(self.unlockAcc)})
        tx_hash = self.cont_dict[name].functions.set(value).transact({'from':self.unlockAcc})
        return tx_hash
    
    def mine(self):
        if not self.web3.eth.mining:
            self.web3.eth.miner.start(1)
        

    def get_status(self, name):
        return self.cont_dict[name].functions.get_status().call({'from':self.unlockAcc})
    
    def send_parameter(self, name, value, sample_number, uploader):
        tx_hash = self.cont_dict[name].functions.send_gradient(value,sample_number,uploader).transact({'from':self.unlockAcc})
        return tx_hash
    
    def get_gradient(self, name):
        res = self.cont_dict[name].functions.get_gradient().call({'from':self.unlockAcc})
        if res!="null":
            self.cont_dict[name].functions.get_gradient().transact({'from':self.unlockAcc})
        return res

    def get_ok(self, name):
        #call 后面要跟from的地址，否则地址为0x000
        res = self.cont_dict[name].functions.ok().call({'from':self.unlockAcc}) 
        return res 
    def get_owner(self, name):
        res = self.cont_dict[name].functions.getOwner().call({'from':self.unlockAcc})
        return res
    
    def register_owner(self, name):
        self.cont_dict[name].functions.ownerRegister().transact({'from':self.unlockAcc})

    def _set_debug_mode(self, name, threshold):
        self.cont_dict[name].functions.debug_set(threshold).transact({'from':self.unlockAcc})

    def is_register(self, name, address):
        res = self.cont_dict[name].functions.isRegister(Web3.to_checksum_address(address)).call()
        return res
    
    # def get_uploader(self, name):
    #     res = self.cont_dict[name].functions.get_uploader().call()
    #     if res!=[]:
    #         self.cont_dict[name].functions.get_uploader().transact({'from':self.unlockAcc})
    #     #['0x5E32Fa2642D1e5afE417A9176707f23b11Af9a7D', '0x5E32Fa2642D1e5afE417A9176707f23b11Af9a7D', '0x5E32Fa2642D1e5afE417A9176707f23b11Af9a7D', '0x5E32Fa2642D1e5afE417A9176707f23b11Af9a7D', '0x5E32Fa2642D1e5afE417A9176707f23b11Af9a7D']
    #     return res
    
    def log_acc(self, name, iteration, uploader, acc):
        res = self.cont_dict[name].functions.log_acc(iteration, uploader, acc).transact({'from': self.unlockAcc})
        return res
    def f_log(self, name):
        res = self.cont_dict[name].events.LogEvent().get_logs(fromBlock=1)
        return res
        # return self.cont_dict[name].events.LogEvent()

    def close(self):
        self.web3.is_connected
        
    # def register_owner(self, name):
    #     self.cont_dict[name].functions.registerNode().transact({'from':})

    # def call_func_with_no_parameter(self, contract_name, func_name):
    #     func = getattr(self.cont_dict[contract_name].functions,func_name)
    #     res = func().transact({'from':Web3.to_checksum_address(self.unlockAcc)})


if __name__ == "__main__":
    # myEth = MyEth(8546)
    # myEth.load_contract("cont","[{\"constant\":true,\"inputs\":[],\"name\":\"value\",\"outputs\":[{\"name\":\"\",\"type\":\"uint256\"}],\"payable\":false,\"stateMutability\":\"view\",\"type\":\"function\"},{\"constant\":false,\"inputs\":[{\"name\":\"v\",\"type\":\"uint256\"}],\"name\":\"set\",\"outputs\":[],\"payable\":false,\"stateMutability\":\"nonpayable\",\"type\":\"function\"},{\"constant\":true,\"inputs\":[],\"name\":\"get\",\"outputs\":[{\"name\":\"\",\"type\":\"uint256\"}],\"payable\":false,\"stateMutability\":\"view\",\"type\":\"function\"}]",r"0x88cb595b04ebe6f0d08ea5e3862761b12376d02b")
    # print(myEth.get("cont"))
    # myEth.unlockAccount(r"0x1dd35298f6ce269459f4982fb0821a07fc7073d3","123")
    # tx_hash=myEth.set("cont",12345)

    # myEth = MyEth(8546)
    # myEth2 = MyEth(8547)
    # # myEth2 = MyEth(8546)
    # names, abis, addresses = MyTools.load_contract_info(r"F:\Code\research\Fedml\MyCode\blockchain\contracts", True)
    # for i in range(names.__len__()):
    #     myEth.load_contract(names[i],abis[i],addresses[i])
    #     myEth2.load_contract(names[i],abis[i],addresses[i])
    # # myEth.unlockAccount(r"0x5e32fa2642d1e5afe417a9176707f23b11af9a7d","123")
    # # print(myEth.call_func_with_no_parameter(names[0]))
    # print(myEth.get("test"))
    # print(myEth2.get("test"))
    # myEth.close()
    # myEth.set("test",7)
    # try:
    #     os.remove('./record.txt')
    # except:
    #     pass
    # with open("./record.txt","a") as f:
    #     f.write("good"+"\n")

    myEth = MyEth(8546)
    names, abis, addresses = MyTools.load_contract_info(r".\blockchain_multi_point\contracts", True)
    for i in range(names.__len__()):
        myEth.load_contract(names[i],abis[i],addresses[i])
    # print(myEth.get_owner("register"))
    myEth.unlockAccount(r"0x2d0814c457d60943c303b7a6473882b88feff614","123")
    # myEth.send_parameter('aggregate',"asdfasdxzvxcvzxc",100)
    # print(myEth.get_gradient('aggregate'))
    # print(myEth.get_uploader('aggregate'))
    # myEth.register_owner("register")
    print(myEth.get_status('aggregate'))
    # myEth.f("test","hah123ahqewrqwa",123345)
    # print(myEth.f_log("reward"))

    # myEth.register_owner()
    # print(myEth.is_register('aggregate',"0x5e32fa2642d1e5afe417a9176707f23b11af9a7d"))
    # print(myEth.get_owner('register'))
    # myEth._set_debug_mode('aggregate',5)
    
    # print(myEth.cont_dict['aggregate'].functions.debug_mode().call({'from':myEth.unlockAcc}))
    
