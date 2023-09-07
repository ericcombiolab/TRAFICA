import os


class Transmit_Server:
    def __init__(self, scp_tool='../expect_scp.exp', server_ip='csr30.comp.hkbu.edu.hk', user_name='csyuxu', pass_word=None, save_folder='./', target_path='/datahome/datasets/ericteam/csyuxu/transmit_test'):
        self.scp_tool = scp_tool
        self.server_ip = server_ip
        self.user_name = user_name
        self.pass_word = pass_word
        self.save_folder = save_folder
        self.destination = target_path
        self.current = None


    def step(self, current_exp): # pack and send
        pack_status, packed_file = self.pack(current_exp)

        if pack_status == 0: # packed
            self.send(current_exp, packed_file)
        else:
            print('error in packing', current_exp)


    def pack(self, current_exp):
        self.current = current_exp.replace('(', '\(').replace(')', '\)')   # "()" must be excluded in filepath in CMD
        packed_file_path = os.path.join(self.save_folder, self.current +'.tar.gz') 
        
        if os.path.exists(packed_file_path):    # The packed .tar.gz file is already existing
            return 0, packed_file_path

        model_package_status = os.system('tar -zcvf ' + packed_file_path + ' -C' + self.save_folder + ' ' + self.current )  # pack by using tar tool
        return model_package_status, packed_file_path


    def send(self, current_exp, packed_file):    
        dest_path = os.path.join(self.destination, current_exp.replace('(', '-').replace(')', '-')+'.tar.gz')
        send_cmd = self.scp_tool + ' ' + self.server_ip +' ' + self.user_name +' '+ self.pass_word +' '+ packed_file +' '+ dest_path
        transmit_status = os.system(send_cmd) 
 
        if transmit_status == 0:
            os.system('rm '+ packed_file)
            #os.system('rm -rf '+ packed_file_path)

    def send_one_file(self, file): 
        file_name = file.split('/')[-1]
        send_cmd = self.scp_tool + ' ' + self.server_ip +' ' + self.user_name +' '+ self.pass_word +' '+ file + ' '+ os.path.join(self.destination, file_name)
        transmit_status = os.system(send_cmd) 
    
