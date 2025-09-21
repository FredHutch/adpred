#!/bin/usr/env python

'''
This package includes functions to help running adpred model on protein
sequences.
'''

__author__ = "Ariel Erijman"
__copyright__ = "Copyright 2020, ADpred project"
__credits__ = ["Ariel Erijman"]
__license__ = "MPL 2.0"
__version__ = "1.2.8"
__maintainer__ = "Ariel Erijman"
__email__ = "aerijman@fredhutch.org"
__status__ = "Dev"


from adpred.lib.utils import *
from uuid import uuid4
import os, string
import resource, time
from subprocess import Popen, PIPE, call

import sys
sys.stderr.write(
'''
    \nIf you wish to use a local installation of psipred, please indicate this,
    by assigning the path to local_psipred (e.g. local_psipred = "~/psipred/run_psipred")\n\n
''')


__methods__ = '''
    * ADPred
    * predict
    * saturated_mutagenesis
        
    * class protein: 
        - predict
        - predict_second_struct
        - saturated_mutagenesis
    '''


# path to a local installation of psipred
local_psipred = None

def calculate_psipred(fasta_name):
    '''
    what it does
    ------------
    predict secondary structure

    parameters
    ----------
        - fasta_name: filename of fasta file

    results
    -------
        -string of secondary structure elements.
    '''    

    if local_psipred is not None:
        command = ['bash', 'local_psipred', fasta_name]
        struct = Popen(p, stdout=PIPE).communicate()[0].decode('utf-8').strip().replace('C','-')
    else:
        struct = get_psipred(fasta_name)

    return struct
    

# initialize session and define model architecture                          
K.clear_session()                                                           
inputs = Input(shape=(30,23,1))                                             
x = Conv2D(29, (6,23), activation=softplus)(inputs)                         
x = Flatten()(x)                                                            
x = Dense(100, activation=softplus, kernel_regularizer=regularizers.l2(0.001))(x)
x = Dropout(0.5)(x)                                                         
x = Dense(30, activation=softplus, kernel_regularizer=regularizers.l2(0.001))(x)
x = Dropout(0.5)(x)                                                         
x = Dense(1)(x)                                                             
output = (Activation('sigmoid'))(x)                                         
ADPred = Model(inputs=inputs, outputs=output)                               
ADPred.compile(optimizer='adam', loss='binary_crossentropy', metrics=[auc]) 
ADPred.load_weights(os.path.dirname(__file__) + '/model/ADpred.h5')                                     
                                                                                

def predict(seq, struct=None):
    '''
    what it does
    ------------
    Assigns to each aminoacid of the sequence the probability of being in a 
    AD region.

    parameters
    ----------
      - sequence: sequence of amino acids (aa). OR uniprot ID! 
        ACHTUNG!! ==> Use a valid id or <name>_<specie> or else you might 
        get the wrong result.
      - second_struct: sequence of secondary elements of each aa in sequence.

    returns
    -------
      - numpy array with probabilities of AD for each aa in the input sequence.

    example:
    -------
    >> sequence = ''.join([ aa[i] for i in np.random.randint(len(aa), size=50)) ])
    >> ss = ''.join([ ss[i] for i in np.random.randint(len(ss), size=50)) ])
    
    >> adpred(aa, ss)
    >> 
    '''
    # providing uniprot-ID
    if np.sum([i in aa for i in seq]) < len(seq):
        seq = identifier2fasta(seq)

    # Could still be that sequence has 'X' so that is not an Id
    # but it's not a proper sequence
    if seq == -1:
        sys.exit("Error: The protein sequence might contain X or other not-allowed characters")

    # Initial guess is that psired webserver will be used
    if struct is None:
        rand_fasta_name = '.' + str(uuid4()) + '.fasta'
        with open(rand_fasta_name, 'w') as fa:    # needed to post to psipred
            fa.write(seq)
        
        # if preferred, use a local installation of psipred
        #if local_psipred is not None:
        #    command = ['bash', 'local_psipred', fasta_name]
        #    struct = Popen(p, stdout=PIPE).communicate()[0].decode('utf-8').strip().replace('C','-')
        #else:
        #    struct = get_psipred(rand_fasta_name)
        struct = calculate_psipred(rand_fasta_name)
    
    # extend adapters for the extremes #####, unless is a 30mer (e.g. in saturated mutagenesis)
    seq = ''.join(['G']*15) + seq + ''.join(['G']*15)
    struct = ''.join(['-']*15) + struct + ''.join(['-']*15)
                                                                                
    # encode for keras and initialize results                                   
    ohe = make_ohe(seq,struct)                                                  
    results = np.zeros(len(seq)-30)                                             
                                                                                
    # roll window of predictions                                                
    for n in range(results.shape[0]):                                           
        results[n] = ADPred.predict(ohe[n:n+30].reshape(1,30,23,1))[0][0]       

    return results


def saturated_mutagenesis(sequence, second_struct, prediction, *args):
    '''
    what is does
    ------------
    Uses mut_analysis to calculate saturated mutagenesis

    parameters
    ----------
        - sequence: Protein sequence
        - second_struct: string of protein secondary structure elements
        - prediction: float - prediction of the corresponding 30mer
        
        - optionals
            - 'second_struct_on_each_mutant': string, second_struct will be computed
              on each mutant.

    returns
    -------
        - 2d array of positions and adpred probabilities for each of the 30 residues
          in the sequence. The order follows the order of the list utils.aa
    '''    

    adpred_results = np.ones(shape=(len(sequence), len(aa))) * prediction
    
    for n_pos, pos in enumerate(sequence):
        seq = list(sequence)  # make a new copy to work with so all other positions are wild type
        
        for n_res, res in enumerate(aa):
            if res == pos:  # don't compute when it's wild type 
                continue  
            else:
                seq[n_pos] = res
                Seq = ''.join(seq)
                if 'second_struct_on_each_mutant' in args:
                    second_struct = get_psipred(Seq)

                ohe = make_ohe(Seq, second_struct).reshape(1,30,23,1)
                adpred_results[n_pos, n_res] = ADPred.predict(ohe)

    return np.flipud(adpred_results.T)




def plot_heatmap(heatmap):

    f,ax = plt.subplots(1, figsize=(20,20))
    im = ax.pcolor(heatmap.T)
    ax.set_xticks(np.arange(30)+0.5)
    ax.set_xticklabels(gcn4.sequence[108:138])
    ax.set_yticks(np.arange(20)+0.5)
    ax.set_yticklabels(adpred.aa)
    plt.colorbar(im, shrink=0.6)


class protein:
    '''
    what is:
    -------
    This class has the following attributes:
        - prot_id: Protein ID (Ideally Uniprot identifier)
        - sequence: Protein sequence
        - second_struct: Secondary structure of protein (from psipred)
        - predictions: ADpred prediction.
        - heatmaps: list of all heatmaps (product of saturated mutagenesis + adpred
          in several 30mers along the protein sequence) available for this protein.
          it's a dictionary where keys are the start position of the 30mer.
    '''
    def __init__(prot, 
                 prot_id = None, 
                 sequence = None, 
                 second_struct = None, 
                 predictions = None,
                 meta_data = None):
        
        assert not (prot_id is None and sequence is None), "Please, provide with a protein " +\
                "ID or a Sequence (sequence + some id that you come up with is also good)" 
       
        prot.prot_id = prot_id

        if sequence == None:
            sequence, meta_data = identifier2fasta(prot_id)
            
        prot.sequence = sequence 
        prot.meta_data = meta_data           
        prot.second_struct = second_struct
        prot.predictions = predictions
        prot.heatmaps = {i:[] for i in range(len(sequence))}



    def predict_second_struct(prot):
        '''
        what it dodes:
        --------------
        predict secondary structure of protein object
        '''
        rand_fasta_name = '.' + str(uuid4()) + '.fasta'
        with open(rand_fasta_name, 'w') as fa:    # needed to post to psipred
            fa.write(prot.sequence)

        # if preferred, use a local installation of psipred
        #if local_psipred is not None:
        #    command = ['bash', 'local_psipred', fasta_name]
        #    prot.second_struct = Popen(p, stdout=PIPE).communicate()[0].decode('utf-8').strip().replace('C','-')
        #else:
        #    prot.second_struct = get_psipred(rand_fasta_name)
        prot.second_struct = calculate_psipred(rand_fasta_name)    

        return None
        

    def predict(prot):
        '''
        use predict method to predict adpred probabilities from sequence
        (see adpred.predict method help)
        '''
        if prot.second_struct is None:
            prot.predict_second_struct()

        prot.predictions = predict(prot.sequence, prot.second_struct)

        return None

    
    def saturated_mutagenesis(prot, start):
        '''
        what it does
        ------------
        Computes saturated_mutagenesis on the protein
        region to compute the saturated mutagenesis. Should be a 30mer.

        parameters
        ----------
            - start: integer. start position of the 30mer in the protein.sequence
            - end: integer. end dposition

        returns
        -------
        2d array (see utils.saturated_mutagenesis help)

        '''
        assert type(start)==int, "start and end are integers!"
        end = start + 30
    
        prot.heatmaps[start] = saturated_mutagenesis(prot.sequence[start:end], 
                                                     prot.second_struct[start:end], 
                                                     prot.predictions[start+15])
        return None

