#!/usr/bin/env bash

# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

usage() {
    echo "Classification demo using public SqueezeNet topology"
    echo "-d name     specify the target device to infer on; CPU, GPU, FPGA or MYRIAD are acceptable. Sample will look for a suitable plugin for device specified"
    echo "-help            print help message"
    exit 1
}

error() {
    local code="${3:-1}"
    if [[ -n "$2" ]];then
        echo "Error on or near line $1: $2; exiting with status ${code}"
    else
        echo "Error on or near line $1; exiting with status ${code}"
    fi
    exit "${code}"
}
trap 'error ${LINENO}' ERR

target="CPU"

# parse command line options
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -h | -help | --help)
    usage
    ;;
    -d)
    target="$2"
    echo target = "${target}"
    shift
    ;;
    -sample-options)
    sampleoptions="$2 $3 $4 $5 $6"
    echo sample-options = "${sampleoptions}"
    shift
    ;;
    *)
    # unknown option
    ;;
esac
shift
done

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if ([ "$target" = "MYRIAD" ] || [ "$target" = "HDDL" ]); then
    # MYRIAD and HDDL support networks with FP16 format only
    target_precision="FP16"
else
    target_precision="FP32"
fi
printf "target_precision = ${target_precision}\n"

models_path="$HOME/openvino_models/models/${target_precision}"
irs_path="$HOME/openvino_models/ir/${target_precision}/"

model_name="squeezenet"
model_version="1.1"
model_type="classification"
model_framework="caffe"
dest_model_proto="${model_name}${model_version}.prototxt"
dest_model_weights="${model_name}${model_version}.caffemodel"

model_dir="${model_type}/${model_name}/${model_version}/${model_framework}"
ir_dir="${irs_path}/${model_dir}"

proto_file_path="${models_path}/${model_dir}/${dest_model_proto}"
weights_file_path="${models_path}/${model_dir}/${dest_model_weights}"

target_image_path="$ROOT_DIR/car.png"

run_again="Then run the script again\n\n"
dashes="\n\n###################################################\n\n"


if [ -e "$ROOT_DIR/../../bin/setupvars.sh" ]; then
    setupvars_path="$ROOT_DIR/../../bin/setupvars.sh"
else
    printf "Error: setupvars.sh is not found\n"
fi

if ! . $setupvars_path ; then
    printf "Unable to run ./setupvars.sh. Please check its presence. ${run_again}"
    exit 1
fi

# Step 1. Download the Caffe model and the prototxt of the model
printf "${dashes}"
printf "\n\nDownloading the Caffe model and the prototxt"

cur_path=$PWD

downloader_path="${INTEL_OPENVINO_DIR}/deployment_tools/tools/model_downloader/downloader.py"

if ! [ -f "${proto_file_path}" ] && ! [ -f "${weights_file_path}" ]; then
    printf "\nRun $downloader_path --name ${model_name}${model_version} --output_dir ${models_path}\n\n"
    $python_binary $downloader_path --name ${model_name}${model_version} --output_dir ${models_path};
else
    printf "\nModels have been loaded previously. Skip loading model step."
    printf "\nModel path: ${proto_file_path}\n"
fi

if [ ! -e "$ir_dir" ]; then
    # Step 2. Configure Model Optimizer
    printf "${dashes}"
    printf "Install Model Optimizer dependencies\n\n"
    cd "${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/install_prerequisites"
    . ./install_prerequisites.sh caffe
    cd $cur_path

    # Step 3. Convert a model with Model Optimizer
    printf "${dashes}"
    printf "Convert a model with Model Optimizer\n\n"

    mo_path="${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py"

    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
    printf "Run $python_binary $mo_path --input_model ${weights_file_path} $ir_dir --data_type $target_precision\n\n"
    $python_binary $mo_path --input_model ${weights_file_path} --output_dir $ir_dir --data_type $target_precision
else
    printf "\n\nTarget folder ${ir_dir} already exists. Skipping IR generation  with Model Optimizer."
    printf "If you want to convert a model again, remove the entire ${ir_dir} folder. ${run_again}"
fi

# Step 4. Build samples
printf "${dashes}"
printf "Build Inference Engine samples\n\n"

OS_PATH=$(uname -m)
NUM_THREADS="-j2"

if [ $OS_PATH == "x86_64" ]; then
  OS_PATH="intel64"
  NUM_THREADS="-j8"
fi

samples_path="${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/samples"
build_dir="$HOME/inference_engine_samples_build"
binaries_dir="${build_dir}/${OS_PATH}/Release"

if [ -e $build_dir/CMakeCache.txt ]; then
	rm -rf $build_dir/CMakeCache.txt
fi
mkdir -p $build_dir
cd $build_dir
cmake -DCMAKE_BUILD_TYPE=Release $samples_path
make $NUM_THREADS classification_sample

# Step 5. Run samples
printf "${dashes}"
printf "Run Inference Engine classification sample\n\n"

cd $binaries_dir

cp -f $ROOT_DIR/${model_name}${model_version}.labels ${ir_dir}/

printf "Run ./classification_sample -d $target -i $target_image_path -m ${ir_dir}/${model_name}${model_version}.xml ${sampleoptions}\n\n"
./classification_sample -d $target -i $target_image_path -m "${ir_dir}/${model_name}${model_version}.xml" ${sampleoptions}

printf "${dashes}"
printf "Demo completed successfully.\n\n"
