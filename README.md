Visual Grounding of Learned Dyanmics Model
============

Running PyFleX example
----------------------

Add and compile PyFleX submodule

    git submodule update --init --recursive
    export PYFLEXROOT=${PWD}/PyFleX-dev
    export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH
    export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
    cd PyFleX-dev/bindings; mkdir build; cd build; cmake ..; make -j

Test PyFleX examples

    python -c "import pyflex; pyflex.main()"
    cd ${PYFLEXROOT}/bindings/examples; python test_RigidFall.py
    

Running the model
----------------------------------------------

Available scripts

    bash scripts/train_RigidFall_dy.sh
    bash scripts/train_FluidIceShake_dy.sh
    bash scripts/eval_RigidFall_dy.sh
    bash scripts/eval_FluidIceShake_dy.sh


