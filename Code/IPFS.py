import ipfshttpclient
import os,time,subprocess
# class IPFS():
#     def __init__(self, init_message="/ip4/127.0.0.1/tcp/5001", cache_path="./cache") -> None:
#         self.client=ipfshttpclient.connect(init_message)
#         self.cache_path=cache_path
#         self.temp_path=os.path.join(cache_path,"temp")

#     def push_file(self, file_path):
#         return self.client.add(file_path)

#     def add(self, s):
#         with open(self.temp_path, "w") as f:
#             f.truncate()
#             f.write(s)
#         return self.push_file(self.temp_path)

#     def load(self, hash):
#         self.client.get(hash)

class IPFS():
    def __init__(self, cache_path="./cache", cache_download="temp_download", cache_upload="temp_upload") -> None:
        self.cache_path=cache_path
        self.temp_download_path=os.path.join(cache_path,cache_download)
        self.temp_upload_path=os.path.join(cache_path,cache_upload)
        
    def push_local_file(self, path):
        command="ipfs  add "+path
        p =  subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8')
        res = p.communicate()[0]
        return res.split(" ")[-9]
    
    def download_loacl_file(self, hash, path):
        command=f"ipfs get {hash} -o "+path
        os.popen(command)
        time.sleep(3)  

    def push_local_str(self, s):
        try:
            os.remove(self.temp_download_path)
        except:
            pass
        with open(self.temp_download_path,"w") as f:
            f.write(s)
        command="ipfs  add "+self.temp_download_path
        p =  subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8')
        res = p.communicate()[0]
        return res.split(" ")[-9]
    def debug():
        command="ipfs add ./gradient_push_copy.txt"
        p =  subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8')
        res = p.communicate()[0]
        return res.split(" ")[-9]
    def _get_local_ipfs_bak(hash):
        command="ipfs cat "+hash
        p =  subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8')
        res = p.communicate()[0]
        begin=0
        for _i in range(0,len(res)):
            if res[_i]=="[":
                begin=_i
                break
        _i = len(res)-1
        while res[_i]!="]":
            _i-=1
        return res[begin:_i+1]
    def debug2():
        command="ipfs get QmPCYDw9S4fuy8d9GrTXaXytVeuWmj9tKAhq7Guy25Ue8h -o ./123"
        os.popen(command)
        # p =  subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8')
        # res = p.communicate()[0]
        # return res.split(" ")[-9]
    def get_local_str(self, hash):
        try:
            os.remove(self.temp_upload_path)
            # pass
        except:
            pass
        command=f"ipfs get {hash} -o "+self.temp_upload_path
        os.popen(command)
        time.sleep(3)
        with open(self.temp_upload_path,"r") as f:
            cont=f.read()
        return cont    

    def ipfs_clear(self):
        command="ipfs repo gc"
        os.popen(command)



if __name__ == "__main__":
    ipfs =  IPFS()
    # print(ipfs.push_local_ipfs("good good study"))
    