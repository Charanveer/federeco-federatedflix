from server import run_server, sample_clients, get_client
from dataset import Dataset
import argparse
from flask import Flask, jsonify
app = Flask(__name__)

trained_model = None
dataset = None
client_dataset= None

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='federeco',
        description='federated recommendation system',
    )

    parser.add_argument('-d', '--dataset', default='movielens', metavar='dataset',
                        choices=['movielens', 'pinterest'],
                        help='which dataset to use, default "movielens"')
    parser.add_argument('-p', '--path', default='pretrained/ncf.h5', metavar='path',
                        help='path where trained model is stored, default "pretrained/ncf.h5"')
    return parser.parse_args()


def main():
    args = parse_arguments()
    # instantiate the dataset based on passed argument
    global dataset
    dataset = Dataset(args.dataset)
    # run the server to load the existing model or train & save a new one
    global trained_model
    trained_model = run_server(dataset, num_clients=20, num_rounds=10, path=args.path)
    # pick random client & generate recommendations for them
    client = sample_clients(dataset, 1)[0]
    print(dataset)
    recommendations = client.generate_recommendation(server_model=trained_model, num_items=dataset.num_items, k=10)
    print('Recommendations for user id: ', client.client_id)
    print(recommendations)
    global client_dataset
    client_dataset = dataset.load_client_train_data()


@app.route('/hello')
def hello():
    return "Hello, World!"

@app.route('/getrandomrec')
def getRandRec():
    print(dataset)
    client = sample_clients(dataset, 1)[0]
    print('Client')
    print(client)
    recommendations = client.generate_recommendation(server_model=trained_model, num_items=dataset.num_items, k=10)
    my_str = ','.join(map(str, recommendations))
    # return jsonify(recommendations)
    return my_str

@app.route('/getrec/<client_id>')
def getRec(client_id):
    print(dataset)
    client = get_client(client_dataset, client_id=client_id)

    print('Client')
    print(client)

    recommendations = client.generate_recommendation(server_model=trained_model, num_items=dataset.num_items, k=10)
    my_str = ','.join(map(str, recommendations))
    # return jsonify(recommendations)
    return my_str

if __name__ == '__main__':
    main()
    app.run(debug=True)
