from fedml.data.MNIST.data_loader import download_mnist,batch_data,read_data
from fedml.data.data_loader import data_server_preprocess,combine_batches
import fedml
from fedml.core.alg_frame.client_trainer import ClientTrainer
import torch.nn as nn
import torch, copy, os, time, json, random
import numpy as np
import logging, math
import IPFS
import blockchain
t1=0.0
class Client:
    def __init__(
        self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model_trainer,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.model_trainer.set_id(client_idx)
    #print(type(self.local_training_data)) --- <class 'list'>
    #print(self.local_training_data[0][0].__len__()) -- 10 -- 训练
    #print(self.local_training_data[0][1].__len__()) -- 10 -- 标签
    #这样改 self.local_training_data[0][1][2] = 1

    def poisoning(self):
        fault_training_data = copy.deepcopy(self.local_training_data)
        for idx in range(fault_training_data.__len__()):
            for idx2 in range(fault_training_data[idx][1].__len__()):
                # fault_training_data[idx][1][idx2] = 9 - fault_training_data[idx][1][idx2]
                fault_training_data[idx][1][idx2] = random.randint(0,9)
        # for idx in range(fault_training_data[1][1].__len__()):
        #     fault_training_data[1][1][idx] = 9 - fault_training_data[1][1][idx]
        self.local_training_data = fault_training_data

        

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics

class MyTrainer(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()
    
    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []

            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                labels = labels.long()
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()
                optimizer.step()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )

                batch_loss.append(loss.item())
            if len(batch_loss) == 0:
                epoch_loss.append(0.0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))


    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                target = target.long()
                loss = criterion(pred, target)  # pylint: disable=E1102

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        return metrics


class BCManager():
    def __init__(self) -> None:
        self.handler=[]
        myEth = blockchain.MyEth(8546)
        # myEth2 = blockchain.MyEth(8547)
        # myEth3 = blockchain.MyEth(8548)
        # myEth4 = blockchain.MyEth(8549)
        self.handler.append(myEth)
        # self.handler.append(myEth2)
        # self.handler.append(myEth3)
        # self.handler.append(myEth4)
        # myEth2 = MyEth(8546)
        names, abis, addresses = blockchain.MyTools.load_contract_info(r"F:\Code\research\Fedml\MyCode\blockchain_test\contracts", True)
        for i in range(names.__len__()):
            for handle in self.handler:
                handle.load_contract(names[i],abis[i],addresses[i])
        self.accounts = []
        self.accounts.append("2d0814c457d60943c303b7a6473882b88feff614")
        # self.accounts.append("225ceaadcf1b0a806a4fae6134a4766d4045f219")
        # self.accounts.append("dfa9ec767bf601e264f8ab2630a25b5c28c488b7")
        # self.accounts.append("f5fc56694e3db2571b2e0409b647f0dfb4870d10")
        
    def get_blockchain_handler(self, client_idx):
        return self.handler[client_idx%self.handler.__len__()]
    
    def maintain_handler(self, client_idx):
        self.handler[client_idx%self.handler.__len__()].unlockAccount(self.accounts[client_idx%self.handler.__len__()],"123")
        

class MyFedAvg():
    def __init__(self, args, device, dataset, model):
        self.device = device
        self.args = args
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ] = dataset

        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.model = model
        self.model_trainer = MyTrainer(self.model, self.args)

        self.rewards = [5.0]*1000
        

        # self.ipfs = IPFS.IPFS()
        # self.bc_manager = BCManager()
        

        self._setup_clients(
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.model_trainer,
        )

    def train(self):
        try:
            os.remove('./record.txt')
        except:
            pass
        print("self.model_trainer = {}".format(self.model_trainer))
        w_global = self.model_trainer.get_model_params()
        # mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        # mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        # mlops.log_round_info(self.args.comm_round, -1)
        delta = 51
        acc_datas = []
        for round_idx in range(self.args.comm_round):

            print("################Communication round : {}".format(round_idx))

            w_locals = []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self._client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            print("client_indexes = " + str(client_indexes))
            
            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx],
                )
                # for iidx in range(0,500):
                #     client_data = self.train_data_local_dict[iidx]
                #     for item in client_data:
                #         print(item[1])
                
                # if random.randint(1,10) <= 8 and client_idx < 200:
                #     print("===================client {} poisoning===================".format(client_idx))
                #     client.poisoning()
                temp = args.epochs
                # if client_idx < 200 and random.randint(1,10) <= 5:
                #     print("===================client {} poisoning===================".format(client_idx))
                #     client.poisoning()
                #     args.epochs=20+int(round_idx/2)


                # client.poisoning()
                # train on new dataset
                # mlops.event("train", event_started=True, event_value="{}_{}".format(str(round_idx), str(idx)))
                print("client {} is training".format(client_idx))
                w = client.train(copy.deepcopy(w_global))
                args.epochs = temp
                # mlops.event("train", event_started=False, event_value="{}_{}".format(str(round_idx), str(idx)))
                # self.logging.info("local weights = " + str(w))
                # try:
                #     os.remove('./cache/weight.pt')
                # except:
                #     pass
                # torch.save(w, './cache/weight.pt')
                # hash = self.ipfs.push_local_file('./cache/weight.pt')
                # bc_handler=self.bc_manager.get_blockchain_handler(client_idx)
                # self.bc_manager.maintain_handler(client_idx)
                # bc_handler.send_parameter('aggregate',hash, client.get_sample_number())
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
            
            acc_locals = []
            client = self.client_list[0]
            for idx in range(w_locals.__len__()):
                num, w = w_locals[idx]
                self.model_trainer.set_model_params(w)
                acc = 0.0
                for idx2 in range(w_locals.__len__()):
                    client.update_local_dataset(0, self.train_data_local_dict[client_indexes[idx2]],self.test_data_local_dict[client_indexes[idx2]],self.train_data_local_num_dict[client_indexes[idx2]])
                    # test data
                    test_local_metrics = client.local_test(True)
                    acc+=(test_local_metrics["test_correct"]/test_local_metrics["test_total"])
                acc /= w_locals.__len__()
                # bc_handler=self.bc_manager.get_blockchain_handler(0)
                # self.bc_manager.maintain_handler(0)
                # bc_handler.log_acc('reward',round_idx, str(client_indexes[idx]), int(acc*100))
                print("client {} acc = {}".format(client_indexes[idx], acc))
                
                acc_locals.append((acc, idx))
            acc_locals = sorted(acc_locals)
            ave_acc=0.0
            for idx in range(1,acc_locals.__len__()-1):
                ave_acc += acc_locals[idx][0]
            # ave_acc *= 100
            ave_acc /= (acc_locals.__len__()-2)
            # ave_acc = int(ave_acc)

            w_locals2 = [] 
            if delta > 2:
                delta -= 1
            for acc, idx in acc_locals:
                loss = ave_acc - acc
                loss *= 100
                flag = 0
                if loss > delta:
                    if loss > delta + 1:
                        print("maliculous client catch!")
                        self.rewards[client_indexes[idx]] -= math.exp(0.15*loss)
                        flag = 1
                    else:
                        self.rewards[client_indexes[idx]] += loss
                        self.rewards[client_indexes[idx]] += 1
                else:
                    loss = -loss
                    if loss > delta + 3:
                        self.rewards[client_indexes[idx]] += math.exp(0.10*loss)
                        self.rewards[client_indexes[idx]] += 1
                    else:
                        self.rewards[client_indexes[idx]] += abs(loss)
                        self.rewards[client_indexes[idx]] += 1
                if flag == 0:
                    w_locals2.append(w_locals[idx])
            # w_locals = w_locals2
            # update global weights
            # mlops.event("agg", event_started=True, event_value=str(round_idx))
            # querry="null"
            # bc_handler=self.bc_manager.get_blockchain_handler(0)
            # while querry=="null":
            #     print("querrying gradients hash")
            #     time.sleep(2)
            #     self.bc_manager.maintain_handler(0)
            #     querry = bc_handler.get_gradient('aggregate')
            # querry = querry.split("|")[:-1]
            # print("successful")
            # print(querry)
            # w_locals=[]
            # uploader_querry = []
            # while uploader_querry == []:
            #     print("querrying uploaders")
            #     time.sleep(2)
            #     uploader_querry = bc_handler.get_uploader('aggregate')
                
            # for idx in range(int(querry.__len__()/2)):
            #     print(int(querry.__len__()/2))
            #     hash = querry[idx*2]
            #     sample_number = int(querry[idx*2+1])
            #     try:
            #         os.remove('./cache/down_weight.pt')
            #     except:
            #         pass
            #     self.ipfs.download_loacl_file(hash, "./cache/down_weight.pt")
            #     time.sleep(1)
            #     w_locals.append((sample_number, torch.load("./cache/down_weight.pt")))
                # time.sleep(1)

            w_global = self._aggregate(w_locals)
            self.model_trainer.set_model_params(w_global)
            # mlops.event("agg", event_started=False, event_value=str(round_idx))

            # test results
            # at last round
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    self._local_test_on_all_clients(round_idx)

            # mlops.log_round_info(self.args.comm_round, round_idx)
            bad = 0
            for i in range(0,200):
                bad += self.rewards[i]
            good = 0
            for i in range(200,500):
                good += self.rewards[i]
            print("good = {}, bad = {}".format(good/300.0,bad/200.0))
            acc_datas.append("good = {}, bad = {}".format(good/300.0,bad/200.0))
        with open ("./record.txt",'a') as f:
            for acc_data in acc_datas:
                print(acc_data)
                f.write(acc_data+"\n")
        # mlops.log_training_finished_status()
        # mlops.log_aggregation_finished_status()

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            # np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        print("client_indexes = %s" % str(client_indexes))
        return client_indexes 


    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        for client_idx in range(self.args.client_num_per_round):
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                model_trainer,
            )
            self.client_list.append(c)

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params
    
    def _local_test_on_all_clients(self, round_idx):
        with open("./record.txt","a") as f:
            logging.info("################local_test_on_all_clients : {}".format(round_idx))
            f.write("################local_test_on_all_clients : {}".format(round_idx)+"\n")
            train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

            test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

            client = self.client_list[0]

            for client_idx in range(self.args.client_num_in_total):
                """
                Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
                the training client number is larger than the testing client number
                """
                if self.test_data_local_dict[client_idx] is None:
                    continue
                client.update_local_dataset(
                    0,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx],
                )
                # train data
                train_local_metrics = client.local_test(False)
                train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
                train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
                train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))
                
                # test data
                test_local_metrics = client.local_test(True)
                test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
                test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
                test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))

            # test on training dataset
            train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
            train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

            # test on test dataset
            test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
            test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])

            stats = {"training_acc": train_acc, "training_loss": train_loss}
            # if self.args.enable_wandb:
            #     wandb.log({"Train/Acc": train_acc, "round": round_idx})
            #     wandb.log({"Train/Loss": train_loss, "round": round_idx})

            # mlops.log({"Train/Acc": train_acc, "round": round_idx})
            # mlops.log({"Train/Loss": train_loss, "round": round_idx})
            logging.info(stats)
            f.write(json.dumps(stats))
            stats = {"test_acc": test_acc, "test_loss": test_loss}
            # if self.args.enable_wandb:
            #     wandb.log({"Test/Acc": test_acc, "round": round_idx})
            #     wandb.log({"Test/Loss": test_loss, "round": round_idx})

            # mlops.log({"Test/Acc": test_acc, "round": round_idx})
            # mlops.log({"Test/Loss": test_loss, "round": round_idx})
            logging.info(stats)
            f.write(json.dumps(stats))

            logging.info("computing time = {}".format(str(time.perf_counter()-t1)))
            f.write("computing time = {}".format(str(time.perf_counter()-t1)))

    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {"test_acc": test_acc, "test_loss": test_loss}
            # if self.args.enable_wandb:
            #     wandb.log({"Test/Acc": test_acc, "round": round_idx})
            #     wandb.log({"Test/Loss": test_loss, "round": round_idx})

            # mlops.log({"Test/Acc": test_acc, "round": round_idx})
            # mlops.log({"Test/Loss": test_loss, "round": round_idx})

        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_pre = test_metrics["test_precision"] / test_metrics["test_total"]
            test_rec = test_metrics["test_recall"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {
                "test_acc": test_acc,
                "test_pre": test_pre,
                "test_rec": test_rec,
                "test_loss": test_loss,
            }
            # if self.args.enable_wandb:
            #     wandb.log({"Test/Acc": test_acc, "round": round_idx})
            #     wandb.log({"Test/Pre": test_pre, "round": round_idx})
            #     wandb.log({"Test/Rec": test_rec, "round": round_idx})
            #     wandb.log({"Test/Loss": test_loss, "round": round_idx})

            # mlops.log({"Test/Acc": test_acc, "round": round_idx})
            # mlops.log({"Test/Pre": test_pre, "round": round_idx})
            # mlops.log({"Test/Rec": test_rec, "round": round_idx})
            # mlops.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        logging.info(stats)




class MyRunner():
    def __init__(self, args, device, dataset, model) -> None:
        self.args = args
        self.device = device
        self.dataset = dataset
        self.model = model
        self.runner=MyFedAvg(args, device, dataset, model)
    
    def run(self):
        self.runner.train()


def load_data(args):
    if args.training_type == "cross_silo" and args.dataset == "cifar10" and hasattr(args, 'synthetic_data_url') and args.synthetic_data_url.find("https") != -1:
        data_server_preprocess(args)
    dataset_name = args.dataset
    # check if the centralized training is enabled
    centralized = True if (args.client_num_in_total == 1 and args.training_type != "cross_silo") else False

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False
    download_mnist(args.data_cache_dir)
    logging.info("load_data. dataset_name = %s" % args.dataset)
    (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = load_partition_data_mnist(
        args,
        args.batch_size,
        train_path=os.path.join(args.data_cache_dir, "MNIST", "train"),
        test_path=os.path.join(args.data_cache_dir, "MNIST", "test"),
    )
    """
    For shallow NN or linear models, 
    we uniformly sample a fraction of clients each round (as the original FedAvg paper)
    """
    args.client_num_in_total = client_num
    if centralized:
        train_data_local_num_dict = {
            0: sum(user_train_data_num for user_train_data_num in train_data_local_num_dict.values())
        }
        train_data_local_dict = {
            0: [batch for cid in sorted(train_data_local_dict.keys()) for batch in train_data_local_dict[cid]]
        }
        test_data_local_dict = {
            0: [batch for cid in sorted(test_data_local_dict.keys()) for batch in test_data_local_dict[cid]]
        }
        args.client_num_in_total = 1

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {
            cid: combine_batches(train_data_local_dict[cid]) for cid in train_data_local_dict.keys()
        }
        test_data_local_dict = {cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()}
        args.batch_size = args_batch_size

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]

    return dataset, class_num

def load_partition_data_mnist(
    args, batch_size, train_path=os.path.join(os.getcwd(), "MNIST", "train"),
        test_path=os.path.join(os.getcwd(), "MNIST", "test")
):
    users, groups, train_data, test_data = read_data(train_path, test_path)

    if len(groups) == 0:
        groups = [None for _ in users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = -1
    logging.info("loading data...")
    step = int(1000/args.client_num_in_total)
    idx = 0
    step_sum = 0
    client_train_data = {'x':[],'y':[]}
    client_test_data = {'x':[],'y':[]}
    for u, g in zip(users, groups):
        flag = 0
        if(idx == step_sum):
            step_sum += step
            user_train_data_num = len(train_data[u]["x"])
            user_test_data_num = len(test_data[u]["x"])
            client_idx += 1
            client_train_data = {'x':[],'y':[]}
            client_test_data = {'x':[],'y':[]}
            flag = 1
        else:
            user_train_data_num += len(train_data[u]["x"])
            user_test_data_num += len(test_data[u]["x"])
        client_train_data['x'] += train_data[u]['x']
        client_test_data['x'] += test_data[u]['x']
        client_train_data['y'] += train_data[u]['y']
        client_test_data['y'] += test_data[u]['y']
        train_data_local_num_dict[client_idx] = user_train_data_num

        if flag == 1:
            # transform to batches
            train_batch = batch_data(args, client_train_data, batch_size)
            test_batch = batch_data(args, client_test_data, batch_size)

            # index using client index
            train_data_local_dict[client_idx] = train_batch
            test_data_local_dict[client_idx] = test_batch
            train_data_global += train_batch
            test_data_global += test_batch
        idx += 1
    logging.info("finished the loading data")
    client_num = client_idx
    class_num = 10

    return (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )

if __name__ == '__main__':
    t1 = time.perf_counter()
    args = fedml.init()

    device = fedml.device.get_device(args)

    dataset, output_dim = load_data(args)

    model = fedml.model.create(args, output_dim)

    runner=MyRunner(args,device,dataset,model)
    runner.run()

    

