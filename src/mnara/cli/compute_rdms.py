import click
import numpy as np
from mnara.pipeline.rdm import compute_rdms, compute_time_rdms

@click.command(context_settings={'show_default': True})
@click.option('--mode', '-m', type=click.Choice(['rdms', 'time']), default='rdms',
              help="'rdms' for static/batch, 'time' for time-resolved")
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=True), help='NPY file or directory')
@click.option('--output', '-o', 'output_path', required=True,
              type=click.Path(), help='Output NPY file')
@click.option('--metric', default='correlation',
              help='Distance metric')
@click.option('--backend', default='numpy',
              help='Computation backend')
@click.option('--n-jobs', default=-1,
              help='CPU jobs for parallel numpy/numba')
@click.option('--parallel/--no-parallel', default=False,
              help='Enable CPU multiprocessing')
@click.option('--dtype', default='float32',
              help='NumPy dtype for outputs')
@click.option('--normalize/--no-normalize', default=False,
              help='Z-score normalize before distances')
@click.option('--return-vector/--no-return-vector', default=False,
              help='Return condensed vectors instead of full matrices')
@click.option('--chunk-size', default=None, type=int,
              help='Chunk size for batch/time loops')
@click.option('--verbose/--no-verbose', default=False,
              help='Show progress bars')
def cli(mode, input_path, output_path, metric, backend,
        n_jobs, parallel, dtype, normalize,
        return_vector, chunk_size, verbose):
    """
    CLI for computing RDMs.

    Delegates to mnara.pipeline.rdm.
    """
    if mode == 'rdms':
        compute_rdms(input_path, output_path,
                     metric=metric, backend=backend,
                     n_jobs=n_jobs, parallel=parallel,
                     dtype=dtype, normalize=normalize,
                     return_vector=return_vector,
                     chunk_size=chunk_size, verbose=verbose)
    else:
        compute_time_rdms(input_path, output_path,
                          metric=metric, backend=backend,
                          n_jobs=n_jobs, parallel=parallel,
                          dtype=dtype, normalize=normalize,
                          return_vector=return_vector,
                          chunk_size=chunk_size, verbose=verbose)

if __name__ == '__main__':
    cli()