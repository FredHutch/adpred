#!/bin/usr/env python

from adpred import ADpred
import sys
import numpy as np

HELP = '''
        using adpred version {}
    
        list of arguments
        -----------------

        -h  | --help 
        -id | --uniprot-id 
        -s  | --sequence 
        -l  | --local-psipred <path_to_"run_psipred">
        -sm | --saturated-mutagenesis (list of start positions separated by comma. Ends are starts+30)
        -o  | --output-prefix (if empty will use protein.id. if prot_id not provided it will be empty)


        examples:
        --------

         - To get only AD predictions:
            run-adpred -id GCN4_YEAST

         - to get also saturated mutagenesis results with AD prediction values: 
            run-adpred -id GCN4_YEAST -sm 108 -o gcn4_satMut108
            run-adpred -id GCN4_YEAST -sm 50,108 -o gcn4_satMut_50-and-108

'''.format(ADpred.__version__)

# help is printed by default
if len(sys.argv)==1 or sys.argv[1] in ["-h","--help"] :
    print(HELP)
    exit(1)

# defaults
start = []
Id, Seq = None, None
out_prefix = None

# user set parameters
for n,arg in enumerate(sys.argv):
    if arg in ["-ID","-id","--uniprot-id","uniprot-ID"]:
        Id = sys.argv[n+1]

    elif arg in ["-s","seq","-Seq","--sequence","--Sequence"]:
        Seq = sys.argv[n+1]

    elif arg in ["-l", "--local-psipred"]:
        ADpred.local_psipred = sys.argv[n+1]

    elif arg in ["-sm", '--saturated-mutagenesis']:
        start = [int(i) for i in sys.argv[n+1].split(",")]
    
    elif arg in ["-o","--output-prefix"]:
        out_prefix = sys.argv[n+1]

# main
if __name__ == '__main__':

    sys.stderr.write("using adpred version {}".format(ADpred.__version__))

    # open file to output results
    if not out_prefix:
        if Id:
            out_prefix = Id 
        elif Seq:
            out_prefix = Seq[:7]
        else:
            sys.stderr.write('You should perovide sequence or uniprot Id..., see --help')
        
    # open output files
    predictions_f = open(out_prefix + '.predictions.csv','w')
    
    if len(start)>0:
        saturated_f = open(out_prefix + '.saturated_mutagenesis.csv','w') 

    # iniitialize protein
    if Id: 
        p = ADpred.protein(prot_id=Id)
        sys.stderr.write('retrieving sequence...')
    elif Seq:
        p = ADpred.protein(sequence=Seq)   
        sys.stderr.write('read sequence ...')

    # predict adpred probabilities
    p.predict()
    sys.stderr.write('calculating secondary structure and adpred...')

    pred_header = "position, aa_id, raw value, smooth1, smooth2"
    pred_body = zip(np.arange(1,len(p.sequence)+1),
                    p.sequence,
                    p.predictions, 
                    np.convolve(p.predictions, np.ones(10)/10, "same"),  
                    np.convolve(p.predictions, np.ones(15)/15, "same"))
                
    pred_body = '\n'.join(["{},{},{},{},{}".format(i[0],i[1],i[2],i[3],i[4]) for i in pred_body])

    predictions_f.write('\n'.join([pred_header, pred_body]))
    
    # compute saturated mutagenesis
    if len(start)>0:
        for i in start:
            p.saturated_mutagenesis(i-1)

            string = [j+','+','.join(list(k.astype(str))) for j,k in zip(ADpred.aa[::-1], p.heatmaps[i-1])]
            saturated_f.write('>' + str(i) + '\n' + '\n'.join(string) + '\n' +\
                             ','+','.join(list(np.arange(i,i+30).astype(str)))+'\n'+\
                             ','+','.join(p.sequence[i-1:i+29])+'\n')
                
    # close written files
    predictions_f.close()

    try:
        saturated_f.close()
    except Exception:
        pass

