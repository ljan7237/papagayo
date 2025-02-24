import click
import numpy as np
import simpsom as sps
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

@click.command()
@click.option("--raw-data", required=True, type=click.Path(exists=True), help="Path to the raw data .npy file.")
@click.option("--new-data", default=None, type=click.Path(exists=True), help="Path to the new data .npy file (optional).")
@click.option("--trained-som", required=True, type=click.Path(exists=True), help="Path to the trained SOM .npy file.")
@click.option("--net-height", default=8, show_default=True, help="Height of the SOM grid.")
@click.option("--net-width", default=8, show_default=True, help="Width of the SOM grid.")
@click.option("--output-file", required=True, type=click.Path(), help="Path to save the output embeddings .npy file.")
def process_som(raw_data, new_data, trained_som, net_height, net_width, output_file):
    """
    Process input data with a trained SOM and generate node difference maps.
    """
    click.echo("Loading raw data...")
    raw_data = np.load(raw_data)
    
    click.echo("Initializing scaler...")
    scaler = MinMaxScaler()
    scaler.fit(raw_data)

    if new_data:
        click.echo(f"Loading new data from {new_data}...")
        new_data = np.load(new_data)
        data = scaler.transform(new_data)
    else:
        click.echo("No new data provided. Processing raw data...")
        data = scaler.transform(raw_data)

    click.echo(f"Loading trained SOM from {trained_som}...")
    trained_som = np.load(trained_som)
    net = sps.SOMNet(data=trained_som, net_height=net_height, net_width=net_width)

    # Initialize an array to store the node difference maps
    all_node_maps = np.zeros((data.shape[0], net_height, net_width))

    click.echo("Processing input data through SOM...")
    
    with tqdm(total=data.shape[0], desc="Processing", unit="sample") as pbar:
        for i in range(data.shape[0]):
            # Find BMU for the input
            bmu_list = net.project(show=False, array=data[i, :].reshape(1, -1), print_out=False)
            bmu, bmu_score = bmu_list[0]

            # Compute node differences
            node_differences = np.zeros((net_height, net_width))
            for j in range(net.net_height):
                for k in range(net.net_width):
                    node_idx = j * net_width + k
                    node_weights = net.node_list[node_idx].weights
                    node_differences[j, k] = np.linalg.norm(node_weights - data[i, :])

            all_node_maps[i, :, :] = node_differences
            pbar.update(1)  # Update progress bar

    click.echo(f"Saving embeddings to {output_file}...")
    np.save(output_file, all_node_maps)
    click.echo("Processing completed!")


if __name__ == "__main__":
    process_som()
# import click
# import numpy as np
# import simpsom as sps
# from sklearn.preprocessing import MinMaxScaler
# from tqdm import tqdm

# @click.command()
# @click.option("--raw-data", required=True, type=click.Path(exists=True), help="Path to the raw data .npy file.")
# @click.option("--new-data", default=None, type=click.Path(exists=True), help="Path to the new data .npy file (optional).")
# @click.option("--trained-som", required=True, type=click.Path(exists=True), help="Path to the trained SOM .npy file.")
# @click.option("--net-height", default=8, show_default=True, help="Height of the SOM grid.")
# @click.option("--net-width", default=8, show_default=True, help="Width of the SOM grid.")
# @click.option("--output-file", required=True, type=click.Path(), help="Path to save the output embeddings .npy file.")
# def process_som(raw_data, new_data, trained_som, net_height, net_width, output_file):
#     """
#     Process input data with a trained SOM and generate node difference maps.
#     """
#     click.echo("Loading raw data...")
#     raw_data = np.load(raw_data)
    
#     click.echo("Initializing scaler...")
#     scaler = MinMaxScaler()
#     scaler.fit(raw_data)

#     if new_data:
#         click.echo(f"Loading new data from {new_data}...")
#         new_data = np.load(new_data)
#         data = scaler.transform(new_data)
#     else:
#         click.echo("No new data provided. Processing raw data...")
#         data = scaler.transform(raw_data)

#     click.echo(f"Loading trained SOM from {trained_som}...")
#     trained_som = np.load(trained_som)
#     net = sps.SOMNet(data=trained_som, net_height=net_height, net_width=net_width)

#     # Initialize an array to store the node difference maps
#     all_node_maps = np.zeros((data.shape[0], net_height, net_width))

#     click.echo("Processing input data through SOM...")
    
#     with tqdm(total=data.shape[0], desc="Processing", unit="sample") as pbar:
#         for i in range(data.shape[0]):
#             # Find BMU for the input
#             bmu_list = net.project(show=False, array=data[i, :].reshape(1, -1), print_out=False)
#             bmu, bmu_score = bmu_list[0]

#             # Compute node differences
#             node_differences = np.zeros((net_height, net_width))
#             for j in range(net.net_height):
#                 for k in range(net.net_width):
#                     node_idx = j * net_width + k
#                     node_weights = net.node_list[node_idx].weights
#                     node_differences[j, k] = np.linalg.norm(node_weights - data[i, :])

#             all_node_maps[i, :, :] = node_differences
#             pbar.update(1)  # Update progress bar

#     click.echo(f"Saving embeddings to {output_file}...")
#     np.save(output_file, all_node_maps)
#     click.echo("Processing completed!")


# if __name__ == "__main__":
#     process_som()