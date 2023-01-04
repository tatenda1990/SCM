import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
import numpy as np
from scipy import stats
from io import BytesIO
import base64

def return_df(f):
    global df #,t,temps
    df = pd.read_csv(f)
    #t = np.array(df[df.columns[0]])
    #temps = [temp for temp in df.columns][1:]
    return df
#f = 'test.csv'
def plot_recoveries (f):
    df = pd.read_csv(f)
    t = np.array(df[df.columns[0]])
    temps = [temp for temp in df.columns][1:]
    fig = plt.figure(figsize = (10,8))
    #fig = Figure()
    ax = fig.subplots()
    #plt.figure(figsize = (10,8))
    for i,temp in enumerate(temps):
        ax.plot(t,df[df.columns[i+1]] , label = str(temp))
        ax.set_xlabel(df.columns[0])
        ax.set_ylabel('Extraction')
        ax.legend()
    #fig.savefig("try.png")
    buf = BytesIO()
    fig.savefig(buf, format = 'png')
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return "data:image/png;base64,{}".format(data)
    #plt.plot(t,df[df.columns[i+1]] , label = str(temp))
    #plt.show()

def surface_chem(df):
    arrs = [np.array(df[col]) for col in df.columns[1:]]
    LH = [(1 - (1-arrs[i])**(1/3)) for i in range(len(arrs))]
    model_linear = "1 - (1-X)**(1/3)"
    return(LH,model_linear)

def diffusion(df):
    arrs = [np.array(df[col]) for col in df.columns[1:]]
    #LH = [(1 - (2/3)*arrs[i]-(1-arrs[i])**(2/3)) for i in range(len(arrs))]
    #model_linear = "1 - (2/3)*X - (1-X)**(2/3)"
    LH = [1 - 3*((1-arrs[i])**(2/3)) + 2*(1-arrs[i]) for i in range(len(arrs))]
    model_linear = "1 - 3*((1-X)**(2/3)) + 2*(1-X)"
    model_name = 'Diffusion'
    return(LH, model_linear)

def mixed(df):
    arrs = [np.array(df[col]) for col in df.columns[1:]]
    LH = [(1 - 2*((1-arrs[i])**(1/3)) + 3*((1-arrs[i])**(2/3))) for i in range(len(arrs))]
    model_linear = "1 - 2*((1-X)**(1/3)) + 3*((1-X)**(2/3))"
    model_name = 'Mixed'
    return(LH, model_linear)

def film(df):
    arrs = [np.array(df[col]) for col in df.columns[1:]]
    LH = [arrs[i] for i in range(len(arrs))]
    model_linear = "X"
    model_name = 'Mixed'
    return(LH, model_linear)

def plot_linear_model(df,LH,linear_model,model_name):
    t = np.array(df[df.columns[0]])
    temps = [temp for temp in df.columns][1:]
    fig = Figure()
    ax = fig.subplots()
    plt.figure(figsize = (10,8))
    for i,entry in enumerate(LH):
        res = stats.linregress(t,entry)
        ax.plot(t,res.intercept + res.slope*t)
        ax.scatter(t,entry)
        ax.set_ylabel(linear_model)
        ax.set_xlabel('t')
        ax.set_title(model_name)
        #print(temps[i],res.slope*1000,res.rvalue)
        #plt.legend(temps) still need to fix this
    buf = BytesIO()
    fig.savefig(buf, format = 'png')
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return "data:image/png;base64,{}".format(data)
        
def plot_linear_kinetic_selected_models (selected_models):
    for model in selected_models:
        LH = model(df)[0]
        model_linear = model(df)[1]
        model_name = model.__name__
        return plot_linear_model(df,LH,model_linear,model_name)
        
def evaluate(df,LH):
    t = np.array(df[df.columns[0]])
    temps = [temp for temp in df.columns][1:]
    res_temps = []
    res_slopes = []
    res_rvalues= []
    for i,entry in enumerate(LH):
        res = stats.linregress(t,entry)
        res_temps.append(temps[i])
        res_slopes.append(res.slope*1000)
        res_rvalues.append(res.rvalue)
        res_df = pd.DataFrame({'Temperature': res_temps, 
                          'k_value': res_slopes,
                          'r_value': res_rvalues})
    return(res_df)

def evaluate_selected_models (selected_models):
    res_collection = []
    for model in selected_models:
        LH = model(df)[0]
        TC_kvalue_rvalue_df = evaluate(df,LH)
        res_tuple = (np.average(TC_kvalue_rvalue_df['r_value']),model)
        res_collection.append(res_tuple)
    res_collection_sorted = sorted(res_collection, reverse = True)
    best_model = res_collection_sorted[0][1]
    rvalues = [entry[0] for entry in res_collection_sorted]
    models_ = [entry[1].__name__ for entry in res_collection_sorted]        
    model_matrix = pd.DataFrame({'Rvalue':rvalues,
                                 'Model':models_})
    model_matrix.index += 1
    print(model_matrix)
    return(best_model, model_matrix)

def best_model(df,selected_models):
    best = evaluate_selected_models(selected_models)[0]
    model_matrix = evaluate_selected_models(selected_models)[1]
    print(model_matrix)
    TC_kvalue_rvalue_df = evaluate(df,best(df)[0])
    TC_kvalue_rvalue_df['lnk'] = np.log(TC_kvalue_rvalue_df['k_value'])
    TC_kvalue_rvalue_df['1000/T'] = 1000/(TC_kvalue_rvalue_df['Temperature'].astype(int)+273.15)
    fig = Figure()
    ax = fig.subplots()
    #plt.figure(figsize = (10,8))
    ax.scatter(TC_kvalue_rvalue_df['1000/T'],TC_kvalue_rvalue_df['lnk'])
    EA_regress = stats.linregress(TC_kvalue_rvalue_df['1000/T'], TC_kvalue_rvalue_df['lnk'])
    ax.plot(TC_kvalue_rvalue_df['1000/T'] , (EA_regress.intercept + (EA_regress.slope * TC_kvalue_rvalue_df['1000/T'])))
    ax.set_xlabel('1000/T (K^-1)')
    ax.set_ylabel('lnk')
    buf = BytesIO()
    fig.savefig(buf, format = 'png')
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    arrhenius_url = "data:image/png;base64,{}".format(data)
    best_rvalue = abs(EA_regress.rvalue)
    print()
    print("Best Model Name: {}".format(best.__name__))
    print("Model RValue: {}".format(np.round(best_rvalue,3)))
    print("Model Slope: {} ".format(np.round(EA_regress.slope,3)))
    print()
    best_model_name = best.__name__
    best_model_r2 = np.round(best_rvalue,3)
    best_model_slope = np.round(EA_regress.slope,3)
    A = np.exp(EA_regress.intercept)
    Ea = -1 * EA_regress.slope * 8.314
    best_model_A = np.round(A,0)
    best_model_Ea = np.round(Ea,2)
    print("Collision Frequency (A): {} ".format(np.round(A,0)))
    print("Activation Energy: {} Kj/mol".format(np.round(Ea,2))) #KJ/mol because we multplied temperature by 1000
    return(model_matrix,best_model_name,best_model_r2,best_model_slope,best_model_A,best_model_Ea,best,arrhenius_url)


selected_models = [surface_chem,diffusion,mixed,film]
#plot_recoveries(f)
#plot_linear_kinetic_selected_models(selected_models)
#best_model(df,selected_models)
#print(df)