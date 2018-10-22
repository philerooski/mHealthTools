import theano
import lasagne
import synapseclient as sc
import multiprocessing as mp
import pandas as pd
import numpy as np
import pickle
import model
import os


def download_test_data(syn):
    q = syn.tableQuery(
            "select healthCode, recordId, 'deviceMotion_walking_outbound.json.items', "
            "'deviceMotion_walking_rest.json.items' from syn10733842")
    outbound_paths = syn.downloadTableColumns(q, "deviceMotion_walking_outbound.json.items")
    rest_paths = syn.downloadTableColumns(q, "deviceMotion_walking_rest.json.items")
    test_table = q.asDataFrame()
    test_table = test_table.groupby("healthCode")
    test_table = test_table.filter(
            lambda r : pd.notnull(r['deviceMotion_walking_outbound.json.items']).all() and \
                       pd.notnull(r['deviceMotion_walking_rest.json.items']).all())
    test_table['outbound_path'] = test_table[
            "deviceMotion_walking_outbound.json.items"].astype(
                    int).astype(str).map(outbound_paths)
    test_table['rest_path'] = test_table[
            "deviceMotion_walking_rest.json.items"].astype(
                    int).astype(str).map(rest_paths)
    return test_table


def data_reader(p):
    df = pd.read_json(p, orient='records')
    data = pd.DataFrame()
    for var in ['x', 'y', 'z']:
        data[var] = [v[var] for v in df.rotationRate.values]
    data = (data - data.mean()) / data.std()
    padding = pd.DataFrame(np.zeros([4000 - len(data), 3]),
                           columns = data.columns)
    padded_data = data.append(padding)
    return padded_data


def get_weights(weights_path):
    weights = []
    for i in range(1, 11):
        with open(os.path.join(weights_path,
                  "fold{}_params_50".format(i)), 'rb') as f:
            w = pickle.load(f, encoding='latin1')
            weights.append(w)
    return weights

# obj should be a dict with keys:
#   weights
#   test_data
#   input_col_name
def predict_using_weights(obj):
    w, test_data, input_col_name = \
            obj['weights'], obj['test_data'], obj['input_col_name']
    prediction_cols = ['healthCode', 'recordId', 'value']
    model_predictions = []
    input_var = theano.tensor.tensor3('input')
    net = model.network(
            input_var = input_var,
            label_var = theano.tensor.ivector('label'),
            shape = (1,3,4000))
    lasagne.layers.set_all_param_values(net, w)
    output_var = lasagne.layers.get_output(net, deterministic=True)
    pred = theano.function([input_var], output_var)
    for i, r in test_data.iterrows():
        healthCode, recordId, path = \
                r['healthCode'], r['recordId'], r[input_col_name]
        data = data_reader(path)
        prediction = pred(np.array([data.values.T], dtype='float32'))
        model_predictions.append(
            [healthCode, recordId, prediction[0][0]])
    model_predictions = pd.DataFrame(
            model_predictions, columns = prediction_cols)
    return(model_predictions)


def predict(test_data, weights_path, input_col_name):
    weights = get_weights(weights_path)
    pool_objects = [{'weights': w, 'test_data': test_data,
        'input_col_name': input_col_name} for w in weights]
    p = mp.Pool(10)
    all_predictions = p.map(predict_using_weights, pool_objects)
    for i in range(len(all_predictions)):
        all_predictions[i]['weight_set'] = i + 1
    all_predictions = pd.concat(all_predictions)
    return all_predictions


def bind_predictions(outbound_predictions, rest_predictions):
    outbound_predictions['assay'] = "outbound"
    rest_predictions['assay'] = "rest"
    all_predictions = outbound_predictions.append(
            rest_predictions, ignore_index = True)
    return(all_predictions)


def main():
    syn = sc.login()
    test_data = download_test_data(syn)
    outbound_predictions = predict(test_data, "weights/outbound/", "outbound_path")
    rest_predictions = predict(test_data, "weights/rest/", "rest_path")
    all_predictions = bind_predictions(outbound_predictions, rest_predictions)
    all_predictions = all_predictions[
            ["healthCode", "recordId", "assay", "weight_set", "value"]]
    all_predictions.to_csv("yfg_predictions.csv", index=False)


if __name__ == "__main__":
    main()


def old_functionality():
    size=int(sys.argv[2])
    model = sys.argv[3]
    params = 'params'

    import pkgutil
    loader = pkgutil.get_importer('model')
    # load network from file in 'models' dir
    model = loader.find_module(model).load_module(model)

    input_var = T.tensor3('input')
    label_var= T.ivector('label')
    shape=(1,3,size)

    net, _, _,_ = model.network(input_var, label_var, shape)

    # load saved parameters from "params"
    with open('params', 'rb') as f:
        import pickle
        params = pickle.load(f)
        lasagne.layers.set_all_param_values(net, params)
        pass

    output_var = lasagne.layers.get_output(net, deterministic=True)
    pred = theano.function([input_var], output_var)


    TEST=open(sys.argv[1],'r')
    for line in TEST:
        line=line.strip()
        table=line.split('\t')
        #image = cv2.imread(table[0], cv2.IMREAD_GRAYSCALE)
        eva.write('%s' % table[0])

    ## get all 500 sequences;
        try:
            data=np.loadtxt(table[1])
            data=data[:,1:]
            data_mean=np.mean(data,axis=0)
            data=(data-data_mean)/np.std(data,axis=0)
            sub_data=np.zeros([3,4000])
            a=len(data)
            if (a<=4000):
                data=data.T
                sub_data[:,0:a]=data
            else:
                data=data.T
                sub_data[:,:]=data[:,0:4000]
            input_pred=[]
            input_pred.append(sub_data)
            input_pred=np.asarray(input_pred,dtype='float32')
            output = pred(input_pred)
            eva.write('\t%.4f' % output)
        except:
            print(line)
            pass


        eva.write('\n')

        pass



