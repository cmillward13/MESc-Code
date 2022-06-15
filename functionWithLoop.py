#Running the RTSA_CALCS function in a loop over the whole database.

#Dependencies
from RTSA_CALCS import * #import function
import pandas as pd
import concurrent.futures
from itertools import repeat
import time

start=time.time()

patient_details=pd.read_excel('./details.xlsx')
right=patient_details.loc[patient_details['RTSA-R']==1][['fname','RTSA-R','RTSA-L']]
left=patient_details.loc[patient_details['RTSA-L']==1][['fname','RTSA-R','RTSA-L']]

rtsa_patients=pd.concat([right,left], ignore_index=True)
files=rtsa_patients['fname']
sides=rtsa_patients['RTSA-R']


def handler(file,side,d,nsa):
        return RTSA_CALCS('./R_Matrices/'+file, side, d, nsa)



if __name__=='__main__':

    with concurrent.futures.ProcessPoolExecutor() as pool:

        Diameters=[38,42]
        NSAs=[135,155]

        #Store Data
        distances=[]
        angles=[]
        times=[]

        for nsa in NSAs:
            for d in Diameters:

                results=pool.map(handler,files,sides,repeat(d),repeat(nsa))

                for  i in results:
                    details=list(i[0:3]) #fname, nsa, cup diameter
                    dist=i[3]
                    t=i[4]
                    #add to lists
                    distances.append(details+dist)
                    times.append(details+t)

    #Put into Dataframes
    names=['Patient','NSA','Diameter','Superior','Inferior','Anterior','Posterior','Centre']
    fname='RTSA_CALCS_PYTHON.xlsx'

    with pd.ExcelWriter(fname, engine='openpyxl') as writer:
        pd.DataFrame(distances,columns=names).to_excel(writer,sheet_name='Sliding Distance')
        pd.DataFrame(times,columns=names).to_excel(writer,sheet_name='Overlap Time')

    end=time.time()
    print(end-start)
