# import libraries used throughout this notebook

import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, log_loss, roc_auc_score, make_scorer
import keras.backend as K
from keras.layers import Input, Dense, Conv2D, Flatten, GlobalMaxPooling2D, AveragePooling2D, MaxPooling2D, Dropout, Activation
from keras.models import Model, model_from_json
from keras.activations import softmax, softplus, softsign, relu
from keras.callbacks import EarlyStopping
from keras import regularizers
import tensorflow as tf
from subprocess import Popen, PIPE, call
from uuid import uuid4

import plotly
import plotly.graph_objs as go
import json, requests, re
from uuid import uuid4 
import pickle
from time import sleep
import os, string


aa = ['R','H','K','D','E','S','T','N','Q','A','V','L','I','M','F' ,'Y', 'W', 'C','G','P']
ss = ['E','H','-'] # list of secondary structure elements


def make_ohe(seq, struct):
    '''
        function returns the data in ohe shape. The columns correspond to the lexicon.
        INPUT: sequence. Sequence of amino acids or secondary structure (ss) elements.
               lexicon. Ordered list of all 20 amino acids or ss elements.
        OUTPUT: ohe_data (shape = (1, len(lexicon))
        e.g. of lexicon for ss: ["E","H","-"] --> beta, alpha, coil

        NOTE: This function can be vectorized since it will constitute a ufunc 
              and the result matrix should have a shape = (len(sequences), len(lexicon))
    '''
    # initialize tensors
    ohe_seq = np.zeros(shape=(len(seq), 20))
    ohe_ss = np.zeros(shape=(len(struct),3))

    # encode sequence and secondary structure
    for n in range(len(seq)):
        ohe_seq[n,aa.index(seq[n])] = 1
        ohe_ss[n, ss.index(struct[n])] = 1

    # join botho tensor 
    ohe = np.vstack([ohe_seq.T, ohe_ss.T]).T #.reshape(1,len(seq),23,1)

    return ohe


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]  # using defaults parameters --> num_thresholds=200
    K.get_session().run(tf.local_variables_initializer())
    return auc


def predict_full(seq, struct, random_id):
    
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
    ADPred.load_weights('../model/ADPred.h5')

    # extend adapters for the extremes
    seq = ''.join(['G']*15) + seq + ''.join(['G']*15)
    struct = ''.join(['-']*15) + struct + ''.join(['-']*15)

    # encode for keras and initialize results
    ohe = make_ohe(seq,struct)
    results = np.zeros(len(seq)-30)

    print(ohe.shape)

    # roll window of predictions
    for n in range(results.shape[0]):
        results[n] = ADPred.predict(ohe[n:n+30].reshape(1,30,23,1))[0][0]
        print(results[n])
    
    # save the csv data
    csv_file = 'predictions/' + random_id + '.csv'
    fasta = 'fastas/' + random_id + '.fasta'
    with open(csv_file,'w') as f:
        f.write('position, amino-acid, adpred-score\n')
        #f.write(','.join([str(i) for i in results]))
        f.write('\n'.join(['{},{},{}'.format(n+1,i,j) for n,(i,j) in enumerate(zip(seq[15:-15],results))]))
    
    #print(random_id+'.csv')


    # save into file the smoothed data
    y = np.array([i if i>0.8 else 0 for i in results])
    results_smooth = np.convolve(y, np.ones(20)/20, "same")    
    csv_file = 'predictions/' + random_id + '_smooth.csv'
    with open(csv_file,'w') as f:
        f.write('position, amino-acid, adpred-score\n')
        f.write('\n'.join(['{},{},{}'.format(n+1,i,j) for n,(i,j) in enumerate(zip(seq[15:-15],results))]))
        #f.write(','.join([str(i) for i in results_smooth]))

    return results, random_id+'.csv', random_id+'_smooth.csv' #, fasta, random_id+'.csv', random_id+'_smooth.csv'


def identifier2fasta(sequence):
    page1 = 'https://www.uniprot.org/uniprot/'+ sequence.replace(' ','').replace('\n','') +'.fasta'
    page2 = 'https://www.uniprot.org/uniprot/?query='+ sequence.replace(' ','').replace('\n','') +'&sort=score'

    # case is a uniprot systematic name 
    try:
        page = requests.get(page1).text 
    except Exception as e:
        print('fasta page could not be downloaded in the first exception',str(e))
        return -1

    # case is a common name (e.g. gcn4) 
    if page[0] == ">":
        return clean_input(page, return_Id=True) #, sequence

    else:
        try:
            page = requests.get(page2).text
            identifier = re.search("<tr id=\".{1,10}\"", page)
            
            if identifier is None: 
                raise ValueError(error_uniprot)
                return -1

            identifier = identifier.group()[7:].replace('"','')
            return clean_input(requests.get('https://www.uniprot.org/uniprot/'+ identifier +'.fasta').text), identifier

        except Exception as e:
            #print('protein name could not be extracted from uniprot site',str(e))
            return -1
    
    return -1


def clean_input(fasta, return_Id=False):
    if fasta[0]==">":
        Input = fasta.split('\n')
        fasta = ''.join(Input[1:])
        Id = Input[0][1:]        
    else:
        Id = "Your protein"

    sequence = fasta.replace('\n','').replace('\r','').replace(' ','').upper()

    if return_Id:
        return sequence, Id    
    else:
        return sequence


def get_psipred(filename, email='rfittipaldi@f1.rum'):

    url = 'https://bioinf.cs.ucl.ac.uk/psipred/api/submission.json'

    payload = {'input_data': (filename, open(filename, 'rb'))}
    data = {'job': 'psipred',
            'submission_name': email,
            'email': 'adprepredictor@gmail.com', }

    r = requests.post(url, data=data, files=payload)

    uid = re.search('{\"UUID\":\"(.*)\",\"sub.*', r.text).group(1)
    submission = 'https://bioinf.cs.ucl.ac.uk/psipred/api/submission/'+uid


    Completed = False
    while not Completed:

        a = requests.get(submission)

        if re.search("&quot;state&quot;: &quot;Complete&quot;,",a.text):
            Completed = True

            horiz = re.search('/submissions/(.*.horiz)&quot', a.text).group(1)
            results = 'https://bioinf.cs.ucl.ac.uk/psipred/api/submissions/' + horiz

            r = requests.get(results)

            if r.status_code == 200:
                unfiltered = r.text
                break
            else:
                Completed = False
        sleep(20)
    
    # extract second struct data from horiz file 
    ss = ''.join([i.group(1) for i in re.finditer('Pred: (.*)\n', unfiltered)]).replace("C","-")

    # free disk space
    os.remove(filename)  

    return ss



def get_psipred_local(sequence):
    fasta_name = 'tmp/' + str(uuid4()) + '.fasta'
    fq = open(fasta_name, 'w')
    fq.write(sequence)
    fq.close()

    p = ['bash', 'psipred/run_psipred_single', fasta_name]
    return Popen(p, stdout=PIPE).communicate()[0].decode('utf-8').strip().replace('C','-')



def create_plot(y_raw, sequence):
    # smooth predictions in two steps
    y = np.array([i if i>0.8 else 0 for i in y_raw])
    y = np.convolve(y, np.ones(20)/20, "same") 

    fig = go.Figure()

    # trace 0
    fig.add_trace( 
        go.Scatter(
            visible=True,
            x=np.arange(len(y_raw)),
            y=y_raw,
            name='raw_data',
            fill='tozeroy',
            fillcolor='rgba(0,0,0,0.1)',
            line={'color':'rgba(0,0,0,0)'}
        )   
    )

    # Add traces, one for each slider step 
    for step in np.arange(1, 52, 1): 
        fig.add_trace( 
            go.Scatter( 
                visible=False, 
                line=dict(color="#00CED1", width=4), 
                name="smooth = " + str(step), 
                x= np.arange(len(y_raw)), 
                y= np.convolve(y_raw, np.ones(step)/step, "same"),    
                #hoverinfo="text", 
                #hovertext=list(sequence)
                text = list(sequence),
                hovertemplate = "Id: %{text}<br>position: %{x}<br>adpred-score: %{y}"
       )    )

    # Make 10th trace visible 
    fig.data[10].visible = True 

    # Create and add slider 
    steps = [] 
    for i in range(len(fig.data)): 
        step = dict( 
            method="restyle", 
            args=["visible", [False] * len(fig.data)], 
        ) 
        step["args"][1][i] = True  # Toggle i'th trace to "visible" 
        step["args"][1][0] = True  # fill raw data always present
        steps.append(step) 
     
    sliders = [dict( 
        active=10, 
        currentvalue={"prefix": "smoothing window: "}, 
        pad={"t": 50}, 
        steps=steps 
    )] 
     
    fig.update_layout( 
        sliders=sliders,
        title={'text': 'ADpred results'}
    )

    

    
    # KEEP THIS
    fig.update_layout(
        yaxis = {'domain':[0, 1]},
        title="", 
        template="plotly_white",
        xaxis = {'title':'residue position'},
        xaxis2 = {'title':'aminoacid',
                  'side':'top',
                  'overlaying':'x',
                  'tickvals':np.arange(len(sequence)),
                  'ticktext':[i for i in sequence],  
                  #'position':1.0,
                  #'anchor':'y'
                }
    )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


# pick this + flanking residues and do mutagenesis --> [ Take the 30mer that ADpred finds as maximum ]
def mutate(sequence, position, to_aa):
    sequence = list(sequence)
    sequence[position] = to_aa
    return ''.join(sequence)


def mut_analysis(sequence, struct, adpred_wt):
    adpred_results = np.ones(shape=(len(sequence), len(aa))) * adpred_wt
    
    for n_pos, pos in enumerate(sequence):
        seq = list(sequence)  # make a new copy to work with so all other positions are wild type
        
        for n_res, res in enumerate(aa):
            if res == pos:  # don't compute when it's wild type 
                continue  
            else:
                seq[n_pos] = res
                Seq = ''.join(seq)
                struct = psipred(Seq)
                ohe = prepare_ohe([Seq, struct]).reshape(1,30,23,1)
                adpred_results[n_pos, n_res] = ADPred.predict(ohe)

    return adpred_results



error_no_mail = 'No email address provided'
error_uniprot = 'The most probably Error is that we couldn\'t resolve your uniprot ID. Please '+\
                'visit <a href="https://www.uniprot.org">Uniprot</a> and look for the ID or the sequence '+\
                'and use it in the <a href="https://adpred.fredhutch.org">home page</a>'
unknown_error = 'Unknown error... Follow the examples. Also, sequences without fasta hearder also works'




'''
import jinja2
def render_without_request(template_name, **template_vars):
    """
    Usage is the same as flask.render_template:

    render_without_request('my_template.html', var1='foo', var2='bar')
    """
    env = jinja2.Environment(
        loader=jinja2.PackageLoader('name.ofmy.package','templates')
    )
    template = env.get_template(template_name)
    return template.render(**template_vars)
'''
