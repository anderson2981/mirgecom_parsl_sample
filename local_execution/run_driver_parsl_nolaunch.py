import parsl
import os
import os.path
from os import path
from parsl.app.app import python_app, bash_app
from parsl.configs.local_threads import config
from parsl.data_provider.files import File

# a configuration to run on local threads
from parsl.config import Config
from parsl.executors.threads import ThreadPoolExecutor

local_threads = Config(
    executors=[
        ThreadPoolExecutor(max_threads=8, label='local_threads')
    ]
)

# a configuration to run locally with pilot jobs
from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from parsl.executors import HighThroughputExecutor
from parsl.providers import LSFProvider
from parsl.providers import SlurmProvider
from parsl.launchers import JsrunLauncher
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_interface

local_htex = HighThroughputExecutor(
    label="local_htex",
    worker_debug=True,
    cores_per_worker=1,
    provider=LocalProvider(
        channel=LocalChannel(),
        init_blocks=1,
        max_blocks=1,
    ),
)

lassen_htex = HighThroughputExecutor(
    label="lassen_htex",
    working_dir='/p/gpfs1/manders/',
    address='lassen.llnl.gov',   # assumes Parsl is running on a login node
    worker_port_range=(50000, 55000),
    worker_debug=True,
    provider=LSFProvider(
        walltime="00:30:00",
        nodes_per_block=1,
        init_blocks=1,
        max_blocks=1,
        scheduler_options='#BSUB -q pdebug',
        worker_init=(
                     'module load module load gcc/7.3.1\n'
                     'module load spectrum-mpi\n'
                     'export XDG_CACHE_HOME="/tmp/$USER/xdg-scratch"\n'
                     'export PYOPENCL_CTX="port:tesla"\n'
                    ),
        project='uiuc',
        cmd_timeout=600
    ),
)

lassen_config = Config(
    executors=[lassen_htex, local_htex],
    strategy=None,
)
print(parsl.__version__)
parsl.set_stream_logger()

parsl.clear()
parsl.load(lassen_config)

# build a string that loads conda correctly
def load_conda():
   return(
            'CONDA_BASE=$(conda info --base)\n'
            'source ${CONDA_BASE}/etc/profile.d/conda.sh\n'
            'conda deactivate\n'
            'conda activate mirge.parsl\n'
         )

@bash_app(executors=['local_htex'])
def build_mirge(execution_string='', stdout='run.stdout', stderr='run.stderr', inputs=[], outputs=[]):
    return('module load gcc; ./buildMirge.sh')

@bash_app(executors=['lassen_htex'])
def run_mirge_lassen(execution_string='', stdout='run.stdout', stderr='run.stderr', inputs=[], outputs=[]):
    return('jsrun -a 1 -n 1 -g 1 ' + execution_string)


# first build MIRGE-Com, this is a local execution
stdout_str = 'buildMirge.stdout'
stderr_str = 'buildMirge.stderr'
completion_file = File(os.path.join(os.getcwd(), 'emirge/buildComplete.txt'))
build = build_mirge(stdout=stdout_str, stderr=stderr_str, outputs=[completion_file])

#print('Result: {}'.format(build.result()))
#print('Done: {}'.format(build.done()))

# first run is just an init, on lassen batch
init_restart_file = File(os.path.join(os.getcwd(), 'restart_data/flame1d-000000-0000.pkl'))
intro_str = 'echo "Running flame1d_init"\n'
conda_str = load_conda()
input_file = File(os.path.join(os.getcwd(), 'run1_params.yaml'))
execution_str = ('python -u -m mpi4py flame_init.py -i {input_file}\n').format(input_file=input_file)
ex_str = intro_str+conda_str+execution_str
stdout_str = 'flame1d_init.stdout'
stderr_str = 'flame1d_init.stdout'
flame_init = run_mirge_lassen(execution_string=ex_str, stdout=stdout_str, stderr=stderr_str, outputs=[init_restart_file])

#print(flame_init.outputs)
print('Done: {}'.format(flame_init.done()))
print('Result: {}'.format(flame_init.result()))
print('Done: {}'.format(flame_init.done()))

# second run is a restart from the init, on lassen batch
run_restart_file = File(os.path.join(os.getcwd(), 'restart_data/flame1d-000000-0000.pkl'))
run_viz_file = File(os.path.join(os.getcwd(), 'restart_data/flame1d-000010.pvtu'))
input_file = File(os.path.join(os.getcwd(), 'run2_params.yaml'))
intro_str = 'echo "Running flame1d_run"\n'
execution_str = (('python -u -m mpi4py flame_run.py -r {restart_file} -i {input_file}\n').
                 format(restart_file=init_restart_file, input_file=input_file))
ex_str = intro_str+conda_str+execution_str
stdout_str = 'flame1d_run1.stdout'
stderr_str = 'flame1d_run1.stdout'
flame_run = run_mirge_lassen(execution_string=ex_str, stdout=stdout_str, stderr=stderr_str, inputs=[input_file, init_restart_file], outputs=[run_viz_file, run_restart_file])

# I get error messages if I do this, even though it finishes correctly...
print(flame_run.outputs[0])
print('Done: {}'.format(flame_run.done()))
print('Result: {}'.format(flame_run.outputs[0].result()))
print('Done: {}'.format(flame_run.done()))
