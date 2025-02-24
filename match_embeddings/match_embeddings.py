import click
import numpy as np
from tqdm import tqdm

@click.command()
@click.option("--new-data", required=True, type=click.Path(exists=True), help="Path to the new song embeddings .npy file.")
@click.option("--birdsong-embeddings", required=True, type=click.Path(exists=True), help="Path to the birdsong embeddings .npy file.")
@click.option("--birdsong-slices", required=True, type=click.Path(exists=True), help="Path to the original birdsong frequency slices .npy file.")
@click.option("--output-matches", type=click.Path(), help="(Optional) Path to save closest match indices .npy file.")
@click.option("--output-slices", required=True, type=click.Path(), help="Path to save reconstructed audio slices .npy file.")
def find_closest_matches(new_data, birdsong_embeddings, birdsong_slices, output_matches, output_slices):
    """
    Find the closest bird song embeddings to the given new song embeddings and reconstruct the corresponding slices.
    """
    click.echo("Loading new song embeddings...")
    N_data = np.load(new_data).astype(np.float32)

    click.echo("Loading birdsong embeddings...")
    M_data = np.load(birdsong_embeddings).astype(np.float32)

    click.echo("Loading birdsong slices...")
    M_slices = np.load(birdsong_slices).astype(np.float32)

    closest_matches = []
    num_slices = N_data.shape[0]

    click.echo("Computing closest matches...")

    with tqdm(total=num_slices, desc="Matching", unit="sample") as pbar:
        for n in range(num_slices):
            distances = np.linalg.norm(N_data[n] - M_data, axis=(1, 2))  # Compute Frobenius norm
            closest_m = np.argmin(distances)  # Find index of closest match
            closest_matches.append(closest_m)
            pbar.update(1)  # Update progress bar

    closest_matches = np.array(closest_matches)

    # Retrieve corresponding slices from M_slices using closest matches
    retrieved_slices = M_slices[closest_matches]

    # Save closest match indices only if output_matches is provided
    if output_matches:
        click.echo(f"Saving closest match indices to {output_matches}...")
        np.save(output_matches, closest_matches)

    click.echo(f"Saving reconstructed audio slices to {output_slices}...")
    np.save(output_slices, retrieved_slices)

    click.echo(f"Processing complete: {len(closest_matches)} matches found.")
    if not output_matches:
        click.echo("Note: Closest match indices were NOT saved because --output-matches was not provided.")

if __name__ == "__main__":
    find_closest_matches()