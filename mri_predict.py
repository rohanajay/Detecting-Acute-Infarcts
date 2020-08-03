from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import time
import logging as log

from openvino.inference_engine import IENetwork, IECore

def build_agrparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input",
                      help="Required. Path to video file or image. 'cam' for capturing video stream from camera",
                      required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                           "kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to labels mapping file", default=None, type=str)
    args.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)

    return parser

    
def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_agrparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    log.info("Creating Inference Engine...")
    ie = IECore()
    #log.info("Creating Inference Engine...")

    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    #log.info("Creating Inference Engine...")

    #Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)
    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    feed_dict = {}
    for blob_name in net.inputs:
        if len(net.inputs[blob_name].shape) == 4:
            input_blob = blob_name
            #print('here')
            #print(net.inputs[blob_name].shape)
        
        else:
            raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                               .format(len(net.inputs[blob_name].shape), blob_name))

    assert len(net.outputs) == 1, "Demo supports only single output topologies"

    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
    exec_net = ie.load_network(network=net, num_requests=2, device_name=args.device)
    
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    
    input_stream = args.input
    assert os.path.isfile(args.input), "Specified input file doesn't exist"
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None
    is_async_mode = True
    request_id = 0
    render_time = time.time()

    #print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    #print("To switch between sync/async modes, press TAB key in the output window")
    result = 0
    while(True):
            
        if is_async_mode:
            img=cv2.imread(input_stream)
            initial_w = img.shape[1]
            initial_h = img.shape[0]
            inf_start = time.time()

            img = cv2.resize(img,(w,h))
            img = img.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            img = img.reshape((n, c, h, w))
            feed_dict[input_blob] = img
            exec_net.start_async(request_id = request_id,inputs = feed_dict)
        if exec_net.requests[request_id].wait(-1) == 0:
            inf_end = time.time()
            det_time = inf_end - inf_start
    
            # Parse detection results of the current request
            res = exec_net.requests[request_id].outputs[out_blob]
            res_list  = list(res[0])
            class_id  = res_list.index(1)
            print('')
            print('Predicted class id : ',class_id)
            predicted_class = labels_map[int(class_id)]
            print('')
            print('Predicted Class name : ',predicted_class)
            '''for obj in res[0][0]:
                        # Draw only objects when probability more than specified threshold
                    
                        class_id = int(obj[1])
                        # Draw box and label\class_id
                        
                        det_label = labels_map[class_id] if labels_map else str(class_id)
                        print('Class predicted is : ', str(class_id))
                        result = str(class_id)'''
            render_start = time.time()
            render_end = time.time()
            render_time = render_end - render_start

        key = cv2.waitKey(0)
        if(key != 0):
            break
    
        
main()
#print('Thankyou very much')
