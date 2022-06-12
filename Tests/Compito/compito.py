#!/usr/bin/env python
# -*- coding: utf-8 -*-


#from contextlib import nullcontext
from curses import erasechar
import json
from random import sample
from tokenize import Floatnumber, String
from types import NoneType
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import codecs


keys=[]
probabilities=[]
key=[]
counter=[]
bool=False

pathToConfigFile="/home/vittorio/TensorFI/confFiles/tests.yaml"

pathToVariables="/usr/local/lib/python2.7/dist-packages/TensorFI/variables.txt"

pathToFolder="/home/vittorio/TensorFI/Tests/Compito"
pathToInstance="/usr/local/lib/python2.7/dist-packages/TensorFI/instance.txt"

#BATCHNORM NON SI PUO FARE, NON CE MODO DIsapere qual e l'istanza precisa della sua ultima adde iniettarla in quella nel caso ci sia piu di un operazione a disposizione
#Mancano EXP e biasADD e batchNorm
#EXP=MUL        BIAS ADD CE         BATCH NORM MI SERVE CALCOLARE IL RISULTATO CON LA CLASSE DELLA FUNZIONE
#                                   TF.NN.BATCHNORMALIZATION
#                                   MEDIA=0.1 VARIANZA=0
#                                   GUARDARE UN MODELLO CON BATCHNORM GUARDARE SU DOCUMENTAZIONE SU TENSORFLOW
#                                   PER LA BIAS ADD 
#                                   MODIFICARE COND PERTURB IN MODO TALE CHE SE DIMENSIONE UNO CHIAMO QUELLE OPERAZIONE SE DIMENSIONE 4 CHIAMO TUTTE LE OPERAZIONE
#                                   QUASI SHATTERGLASS COME SHUTTERGLASS IDENTICO
         #                         
        # PER LA OPS ADD PROVIAMO IL DEBUGGING


         # PER QUANTO RIGUARDA BATCHNORM
         # 
         #  UNCATHEGORIZED E UN RANDOM SEMPLICE ESTRAGGO A CASO NELLA FEATUREMAP E METTO NELLA STESSA E GUARDO SE SULLA STESSA FEATURE MAP LO METTO LI RANDOMICO INVECE SE E LUNGA 3 LO METTO SULLA STESSA FEATURE MA SICURAMENTE SULLA STESSA FEATUREMAP
         #      SCRIVERE LA RELAZIONE 
         #-SCENARIO GENERALE SULLE RETI CON CONST
         # - TENSORFI NON CORRISPONDE A IL REALE EFFETTO FISICO AVREBBE UN GUASTO,IL QUA
         # IN PASSATO E STATO FATTO TUTTO UNO STUDIO S
        # INTEGRARE IN TENSORFI I MODELLI D'ERRORE 

        # - come e fatto tensorflow a livello di software mostrare graficamente cosa sono dovuo ad andare a toccare per la 
        # come ho relaizzato i vari pezzi con esempi significati
        # - RACCONTA IN MANIERA CHE CHI VEDE IL VOTO MI 
        #


        # BLOCK SIGNIFICA GLI INDICI  CON 0,1 INDICANO L'OFFSET DEL BLOCCO QUINDI 16 POSTI DOPO
        # STESSA COSA PER BLOCK E FEATURE MAP BLOCK CE ANCHE L'INDICE DELLA FEATURE MAP 

         #PROBLEMI: In dynamic instaces funziona ma inietta anche una add che non dovrebbe iniettare, risolto semplicemente se c'è una add non la inietto ma 
         #          per quanto riguarda la add comunque ho dei problemi non so come gestirla
input1= input("Scegli l'operazione da iniettare: ADD=1  CONVULUTION=2   DIV=3   RELU=4   MUL=5   SIGMOID=6   BAISADD=7  EXP=8\n")
if(input1==1):
    op= " - ADD=1.0"
    pathToValue= pathToFolder + "/models/S1_add/value_analysis.txt"
    pathToModel=pathToFolder + "/models/S1_add/S1_add_spatial_model.json"
    pathToCount=pathToFolder + "/models/S1_add/add_S1_anomalies_count.json"
elif(input1==2):
    op=" - CONV2D=1.0"
    pathToValue=pathToFolder + "/models/S1_convolution/value_analysis.txt"
    pathToModel=pathToFolder + "/models/S1_convolution/S1_convolution_spatial_model.json"
    pathToCount=pathToFolder + "/models/S1_convolution/convolution_S1_anomalies_count.json"
elif(input1==3):
    op="- SUB=1.0"
    pathToValue=pathToFolder + "/models/S1_div/value_analysis.txt"
    pathToModel=pathToFolder + "/models/S1_div/S1_div_spatial_model.json"
    pathToCount=pathToFolder + "/models/S1_div/div_S1_anomalies_count.json"
elif(input1==4):
    op="- RELU=1.0"
    pathToValue=pathToFolder + "/models/S1_leaky_relu/value_analysis.txt"
    pathToModel=pathToFolder + "/models/S1_leaky_relu/S1_leaky_relu_spatial_model.json"
    pathToCount=pathToFolder + "/models/S1_leaky_relu/leaky_relu_S1_anomalies_count.json"
elif(input1==5):
    op="- MUL=1.0"
    pathToValue=pathToFolder + "/models/S1_mul/value_analysis.txt"
    pathToModel=pathToFolder + "/models/S1_mul/S1_mul_spatial_model.json"
    pathToCount=pathToFolder + "/models/S1_mul/mul_S1_anomalies_count.json"
elif(input1 ==6):
    op=" - SIGMOID=1.0"
    pathToValue=pathToFolder + "/models/S1_sigmoid/value_analysis.txt"
    pathToModel=pathToFolder + "/models/S1_sigmoid/S1_sigmoid_spatial_model.json"
    pathToCount=pathToFolder + "/models/S1_sigmoid/sigmoid_S1_anomalies_count.json"
elif(input1==7):
    op=" - BIASADD=1.0"
    pathToValue=pathToFolder + "/models/S1_biasadd/value_analysis.txt"
    pathToModel=pathToFolder + "/models/S1_biasadd/S1_biasadd_spatial_model.json"
    pathToCount=pathToFolder + "/models/S1_biasadd/biasadd_S1_anomalies_count.json"
elif(input1==8):
    op=" - MATMUL=1.0"
    pathToValue=pathToFolder + "/models/S1_exp/value_analysis.txt"
    pathToModel=pathToFolder + "/models/S1_exp/S1_exp_spatial_model.json"
    pathToCount=pathToFolder + "/models/S1_exp/exp_S1_anomalies_count.json"

g=open(pathToInstance,"w")
instance= input("Scegli l'istanza da iniettare\n")
g.write(str(instance))
global MAX
global RANDOM
probabilities=[]
values=[1,2,3,4]
#Scelgo il valore da iniettare
with open(pathToValue) as inputData:
  inputData.readline()
  Data= inputData.readline()
  data=Data.split("[-1, 1]: ")
  data=data[1].split("\n")
  probabilities.append(float(data[0]))
  Data= inputData.readline()
  data=Data.split("Others: ")
  data=data[1].split("\n")
  probabilities.append(float(data[0]))
  Data= inputData.readline()
  data=Data.split("NaN: ")
  data=data[1].split("\n")
  probabilities.append(float(data[0]))
  Data= inputData.readline()
  data=Data.split("Zeros: ")
  data=data[1].split("\n")
  probabilities.append(float(data[0]))
  Data= inputData.readline()
  data=Data.split("Valid: ")
  data=data[1].split("\n")
  probabilities[0]=(float(data[0])*probabilities[0])
  probabilities[1]=(1-sum(probabilities))+probabilities[1]
  #print(sum(probabilities))
  #print(probabilities)
  
myRV = stats.rv_discrete(name="MyRandomVar", values=(values, probabilities))
value= myRV.rvs()



if(value==1):
    value=np.random.uniform(low=-1 , high=1)
elif(value==2):
    value=np.random.randint(low=100000)     #Scelto un valore tra 0 e low intero
elif(value==3):
    value=""
elif(value==4):
    value=0
print(value)
with open(pathToCount) as inputData:
  Data=json.load(inputData)

probabilities=[]
keys=[]
j=0

with open(pathToCount) as inputData:
  Data=json.load(inputData)


for i in range(len(Data.keys())):
    probabilities.append(float(Data.values()[i][1]))
for i in Data.keys():
    keys.append(int(i))

myRV = stats.rv_discrete(name="MyRandomVar", values=(keys, probabilities))
cardinalita = myRV.rvs()
cardinalita=cardinalita
cardinalita=str(cardinalita)
print(cardinalita)


#scelgo l'operazione
with open(pathToModel) as inputData:
  Data=json.load(inputData)

keys=[]
probabilities=[]
key=[]
counter=[]
if(cardinalita=="1"):
      print("cardinalita singola")
      g=open(pathToConfigFile,"w")
      g.truncate()
      g.write("ScalarFaultType: None \nTensorFaultType: singleFlip\n\nOps:\n " +op + "\n\n")
      g.close()
      g=open(pathToVariables,"w")
      g.write(str(value))
else:
    keys=[]
    probabilities=[]
    for i in Data[cardinalita]["FF"].values():
        probabilities.append(i)
    for i in Data[cardinalita]["FF"].keys():
        keys.append(int(i))
    
    myRV = stats.rv_discrete(name="MyRandomVar", values=(keys, probabilities))
    operation= myRV.rvs()
    operation=str(operation)
    print(operation)

    
    keys=[]
    probabilities=[]
    key=[]
    counter=[]
    if(operation=="0"):
        print("operazione stessa riga\n")
        g=open(pathToConfigFile,"w")
        g.truncate()
        g.write("ScalarFaultType: None \nTensorFaultType: rawbitFlip-tensor\n\nOps:\n " +op+ "\n\n")
        g.close()
        dictionary=Data[cardinalita]["PF"][operation]
        print(dictionary)
        MAX=dictionary["MAX"]
        RANDOM=dictionary["RANDOM"]
        keys=[]
        probabilities=[]
        key=[]
        counter=[]
        if(dictionary.get("RANDOM")==0):
            bool=True
        for i in dictionary.values():
            if(i!=dictionary.get("MAX")):
               
                probabilities.append(float(i))

        for i in dictionary.keys():
            if(i!="MAX"):
                if(bool):
                    if(i!="RANDOM"):
                        keys.append(i)
                        print(type(keys))
                else:
                    keys.append(i)
                    print(type(keys))
        for i in range(len(keys)):
            counter.append(i)
            i=i+1
        myRV = stats.rv_discrete(name="MyRandomVar", values=(counter, probabilities))
        key= myRV.rvs()
        if(keys[key]=="RANDOM"):
            g=open(pathToVariables,"w")
            g.truncate()
            g.write(str(value) +"\n" +str(cardinalita)+"\n"+"RANDOM\n"+str(MAX))
            print("scrittoRANDOM")
        else:
            g=open(pathToVariables,"w")
            g.truncate()
            g.write(str(value) + "\n"+str(cardinalita)+"\n"+str(keys[key]))
            print("scritto")


    if(operation=="1"):
        print("operazione stessa colonna\n")
        dictionary=Data[cardinalita]["PF"][operation]
        print(dictionary)
        MAX=dictionary["MAX"]
        RANDOM=dictionary["RANDOM"]
        g=open(pathToConfigFile,"w")
        g.write("ScalarFaultType: None \nTensorFaultType: columnbitFlip-tensor\n\nOps:\n "+op+"\n\n")
        g.close()
        if(dictionary.get("RANDOM")==0):
            bool=True
        for i in dictionary.values():
            if(i!=dictionary.get("MAX")):
                probabilities.append(float(i))
        for i in dictionary.keys():
            if(i!="MAX"):
                if(bool):
                    if(i!="RANDOM"):
                        keys.append(i)
                        print(type(keys))
                else:
                    keys.append(i)
                    print(type(keys))
            print(type(keys))
        for i in range(len(keys)):
            counter.append(i)
            i=i+1
        myRV = stats.rv_discrete(name="MyRandomVar", values=(counter, probabilities))
        key= myRV.rvs()
        print(keys[key])
        if(keys[key]=="RANDOM"):
            g=open(pathToVariables,"w")
            g.truncate()
            g.write(str(value) +"\n" +str(cardinalita)+"\n"+"RANDOM\n"+str(MAX))
            print("scrittoRANDOM")
        else:
            g=open(pathToVariables,"w")
            g.truncate()
            g.write(str(value) + "\n"+str(cardinalita)+"\n"+str(keys[key]))
            print("scritto")

    if(operation=="2"):
        print("operazione block\n")
        dictionary=Data[cardinalita]["PF"][operation]
        print(dictionary)
        MAX=dictionary["MAX"]
        RANDOM=dictionary["RANDOM"]
        g=open(pathToConfigFile,"w")
        g.write("ScalarFaultType: None \nTensorFaultType: blockSameFeature\n\nOps:\n "+op+"\n\n")
        g.close()
        if(dictionary.get("RANDOM")==0):
            bool=True
        for i in dictionary.values():
            if(i!=dictionary.get("MAX")):
                probabilities.append(float(i))
        for i in dictionary.keys():
            if(i!="MAX"):
                if(bool):
                    if(i!="RANDOM"):
                        keys.append(i)
                        print(type(keys))
                else:
                    keys.append(i)
                    print(type(keys))
            print(type(keys))
        for i in range(len(keys)):
            counter.append(i)
            i=i+1
        myRV = stats.rv_discrete(name="MyRandomVar", values=(counter, probabilities))
        key= myRV.rvs()
        print(keys[key])
        if(keys[key]=="RANDOM"):
            g=open(pathToVariables,"w")
            g.truncate()
            g.write(str(value) +"\n" +str(cardinalita)+"\n"+"RANDOM\n"+str(MAX))
            print("scrittoRANDOM")
        else:
            g=open(pathToVariables,"w")
            g.truncate()
            g.write(str(value) + "\n"+str(cardinalita)+"\n"+str(keys[key]))
            print("scritto")

    if(operation=='3'):
        print("operazione RANDOM")
        dictionary=Data[cardinalita]["PF"][operation]
        print(dictionary)
        MAX=dictionary["MAX"]
        RANDOM=dictionary["RANDOM"]
        g=open(pathToConfigFile,"w")
        g.write("ScalarFaultType: None \nTensorFaultType: sameFeatureRandom\n\nOps:\n "+op+"\n\n")
        g.close()
        if(dictionary.get("RANDOM")==0):
            bool=True
        for i in dictionary.values():
            if(i!=dictionary.get("MAX")):
                probabilities.append(float(i))
        for i in dictionary.keys():
            if(i!="MAX"):
                if(bool):
                    if(i!="RANDOM"):
                        keys.append(i)
                        print(type(keys))
                else:
                    keys.append(i)
                    print(type(keys))
        for i in range(len(keys)):
            counter.append(i)
            i=i+1
        myRV = stats.rv_discrete(name="MyRandomVar", values=(counter, probabilities))
        key= myRV.rvs()
        print(keys[key])
        if(keys[key]=="RANDOM"):
            g=open(pathToVariables,"w")
            g.truncate()
            g.write(str(value) +"\n" +str(cardinalita)+"\n"+"RANDOM\n"+str(MAX))
            print("scrittoRANDOM")
        else:
            g=open(pathToVariables,"w")
            g.truncate()
            g.write(str(value) + "\n"+str(cardinalita)+"\n"+str(keys[key]))
            print("scritto")

    if(operation=='4'):
        print("Operazione RAndom")
        dictionary=Data[cardinalita]["PF"][operation]
        print(dictionary)
        MAX=dictionary["MAX"]
        RANDOM=dictionary["RANDOM"]
        g=open(pathToConfigFile,"w")
        g.write("ScalarFaultType: None \nTensorFaultType: bulletWake\n\nOps:\n "+op+"\n\n")
        g.close()
        if(dictionary.get("RANDOM")==0):
                bool=True
        for i in dictionary.values():
            if(i!=dictionary.get("MAX")):
                probabilities.append(float(i))
        for i in dictionary.keys():
            if(i!="MAX"):
                if(bool):
                    if(i!="RANDOM"):
                        keys.append(i)
                        print(bool)
                else:
                    keys.append(i)
                    print(bool)
        for i in range(len(keys)):
            counter.append(i)
            i=i+1
        myRV = stats.rv_discrete(name="MyRandomVar", values=(counter, probabilities))
        key= myRV.rvs()
        print(keys[key])
        if(keys[key]=="RANDOM"):
            g=open(pathToVariables,"w")
            g.truncate()
            g.write(str(value) +"\n" +str(cardinalita)+"\n"+"RANDOM\n"+str(MAX))
            print("scrittoRANDOM")
        else:
            g=open(pathToVariables,"w")
            g.truncate()
            g.write(str(value) + "\n"+str(cardinalita)+"\n"+str(keys[key]))
            print("scritto")

    if(operation=="5"):
        print("Operazione bullet wake")
        dictionary=Data[cardinalita]["PF"][operation]
        print(dictionary)
        MAX=dictionary["MAX"]
        RANDOM=dictionary["RANDOM"]
        g=open(pathToConfigFile,"w")
        g.write("ScalarFaultType: None \nTensorFaultType: blockDifferentFeature\n\nOps:\n "+op+"\n\n")
        g.close()
        if(dictionary.get("RANDOM")==0):
                bool=True
        for i in dictionary.values():
            if(i!=dictionary.get("MAX")):
                probabilities.append(float(i))
        for i in dictionary.keys():
            if(i!="MAX"):
                if(bool):
                    if(i!="RANDOM"):
                        keys.append(i)
                        print(bool)
                else:
                    keys.append(i)
                    print(bool)
        for i in range(len(keys)):
            counter.append(i)
            i=i+1
        myRV = stats.rv_discrete(name="MyRandomVar", values=(counter, probabilities))
        key= myRV.rvs()
        print(keys[key])
        if(keys[key]=="RANDOM"):
            g=open(pathToVariables,"w")
            g.truncate()
            g.write(str(value) +"\n" +str(cardinalita)+"\n"+"RANDOM\n"+str(MAX))
            print("scrittoRANDOM")
        else:
            g=open(pathToVariables,"w")
            g.truncate()
            g.write(str(value) + "\n"+str(cardinalita)+"\n"+str(keys[key]))
            print("scritto")
    

    if(operation=="6" or operation=="7"):
        print("Operazione shutterGlass")
        dictionary=Data[cardinalita]["PF"][operation]
        print(dictionary)
        MAX=dictionary["MAX"]
        RANDOM=dictionary["RANDOM"]
        g=open(pathToConfigFile,"w")
        g.write("ScalarFaultType: None \nTensorFaultType: shutterGlass\n\nOps:\n "+op+"\n\n")
        g.close()
        if(dictionary.get("RANDOM")==0):
                bool=True
        for i in dictionary.values():
            if(i!=dictionary.get("MAX")):
                probabilities.append(float(i))
        for i in dictionary.keys():
            if(i!="MAX"):
                if(bool):
                    if(i!="RANDOM"):
                        keys.append(i)
                else:
                    keys.append(i)
        for i in range(len(keys)):
            counter.append(i)
            i=i+1
        myRV = stats.rv_discrete(name="MyRandomVar", values=(counter, probabilities))
        key= myRV.rvs()
        if(keys[key]=="RANDOM"):
            if(len(probabilities)==1):
                print("La probabilita di random e una, non gestiamo questo tipo casa, runnare di nuovo il programma")
                quit()
        while(keys[key]=="RANDOM"):
            myRV = stats.rv_discrete(name="MyRandomVar", values=(counter, probabilities))
            key= myRV.rvs()
            print(keys[key])
        g=open(pathToVariables,"w")
        g.truncate()
        g.write(str(value) + "\n"+str(cardinalita)+"\n"+str(keys[key]))
        print("scritto")


    if(operation=="8"):
        print("Operazione uncategorized")
        dictionary=Data[cardinalita]["PF"][operation]
        print(dictionary)
        MAX=dictionary["MAX"]
        RANDOM=dictionary["RANDOM"]
        g=open(pathToConfigFile,"w")
        g.write("ScalarFaultType: None \nTensorFaultType: uncategorize\n\nOps:\n "+op+"\n\n")
        g.close()
        if(dictionary.get("RANDOM")==0):
                bool=True
        for i in dictionary.values():
            if(i!=dictionary.get("MAX")):
                probabilities.append(float(i))
        for i in dictionary.keys():
            if(i!="MAX"):
                if(bool):
                    if(i!="RANDOM"):
                        keys.append(i)
                else:
                    keys.append(i)
        for i in range(len(keys)):
            counter.append(i)
            i=i+1
        myRV = stats.rv_discrete(name="MyRandomVar", values=(counter, probabilities))
        key= myRV.rvs()
        if(keys[key]=="RANDOM"):
            if(len(probabilities)==1):
                print("La probabilita di random e una, non gestiamo questo tipo casa, runnare di nuovo il programma")
                quit()
        while(keys[key]=="RANDOM"):
            myRV = stats.rv_discrete(name="MyRandomVar", values=(counter, probabilities))
            key= myRV.rvs()
            print(keys[key])
        g=open(pathToVariables,"w")
        g.truncate()
        g.write(str(value) + "\n"+str(cardinalita)+"\n"+str(keys[key]))
        print("scritto")



    #import TensorFI.fiConfig as fi
    #pathToConfig="/home/vittorio/TensorFI/confFiles/tests.yaml"
    #g=open(pathToConfig,"r")
    #lines=g.readlines()
    #g.close()
    #g=open(pathToConfig,"w")
    #counter=0
    #for i in lines:
    #    counter=counter+1
    #    if(counter==6):
    #      break
    #    g.write(i)
    #operation=[]
    #g.write("\nInstances:\n")
   
    #for n in tf.get_default_graph().as_graph_def().node:
    
        #string=str(n.op)
        #for f in fi.Ops:
         # if(f.value==string.upper()):
          #  #print(str(f.value) + " \n\n")
           # operation.append(f.value)
    #analized=[]
    #for op in operation:
     #   if(analized.count(op)==0):
           # print( str(op) +" = "+ str(operation.count(op)))
      #      g.write(" - "+str(op) +" = "+ str(operation.count(op))+"\n")
       #     analized.append(op)
    #g.write("\nInjectMode: "+ str("dynamicInstance"))        
    #g.close()