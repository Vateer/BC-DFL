@echo off
start powershell.exe -noexit "ipfs daemon"

start powershell.exe -noexit "geth  -datadir ./data1/ --networkid 88 --http --http.addr=0.0.0.0 --http.port=8546 --http.api 'web3,eth,debug,personal,net' --allow-insecure-unlock --nodiscover console"

start powershell.exe -noexit "geth  -datadir ./data2/ --networkid 88 --http --http.addr=0.0.0.0 --http.port=8547 --authrpc.port=8547 --port 30304 --http.api 'web3,eth,debug,personal,net' --allow-insecure-unlock --nodiscover --ipcdisable console"

start powershell.exe -noexit "geth  -datadir ./data3/ --networkid 88 --http --http.addr=0.0.0.0 --http.port=8548 --authrpc.port=8548 --port 30305 --http.api 'web3,eth,debug,personal,net' --allow-insecure-unlock --nodiscover --ipcdisable console"

start powershell.exe -noexit "geth  -datadir ./data4/ --networkid 88 --http --http.addr=0.0.0.0 --http.port=8549 --authrpc.port=8549 --port 30306 --http.api 'web3,eth,debug,personal,net' --allow-insecure-unlock --nodiscover --ipcdisable console"
