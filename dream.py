from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format
import matplotlib.pyplot as plot

# TODO get CUDA running again and run caffe on GPU!!
import caffe

def show_img(im):
    im = np.uint8(np.clip(im, 0, 255))
    plot.imshow(im, interpolation = 'nearest')
    plot.show()


# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def objective_L2(dst):
    dst.diff[:] = dst.data

def make_step(
        net,
        step_size=1.5,
        end='inception_4c/output',
        jitter=32,
        clip=True,
        objective=objective_L2
        ):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

def deepdream(
        net,
        base_img,
        iter_n = 10,
        octave_n = 4,
        octave_scale = 1.4,
        end = 'inception_4c/output',
        clip = True,
        save_step = False,
        **step_params
        ):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))

        src = net.blobs['data']
        detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
        for octave, octave_base in enumerate(octaves[::-1]):
            h, w = octave_base.shape[-2:]
            if octave > 0:
                # upscale details from the previous octave
                h1, w1 = detail.shape[-2:]
                detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

            src.reshape(1,3,h,w) # resize the network's input image size
            src.data[0] = octave_base + detail
            for i in xrange(iter_n):
                make_step(net, end=end, clip=clip, **step_params)

                # visualization
                vis = deprocess(net, src.data[0])
                if not clip: # adjust image contrast if clipping is disabled
                    vis = vis*(255.0/np.percentile(vis, 99.98))
                if save_step:
                    path = "experiment/" + end + "_octave_" + str(octave) + "iteration_"
                    path += str(i) + ".jpg"
                    PIL.Image.fromarray( np.uint8(vis) ).save(path)
                print octave, i, end, vis.shape

        # extract details produced on the current octave
        detail = src.data[0] - octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])


def deepdream_single_frame(net, img):
    print net.blobs.keys()
    #__= deepdream(net, img)
    _=deepdream(net, img, end='inception_3b/5x5_reduce', save_step = True)


def deepdream_multi_frame(net, img):
    frame = img
    frame_i = 0
    h, w = frame.shape[:2]
    s = 0.05 # scale coefficient
    # TODO more iteration
    for i in xrange(10):
        # TODO plat with octaves and iterations and different end layers!
        frame = deepdream(
                net,
                frame,
                end='inception_3b/5x5_reduce',
                iter_n = 10,
                octave_n = 2
                )
        PIL.Image.fromarray(np.uint8(frame)).save("frames/%04d.jpg"%frame_i)
        frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
        frame_i += 1


def control_the_dream(net):
    guide = np.float32(PIL.Image.open('flowers.jpg'))
    end = 'inception_3b/output'
    h, w = guide.shape[:2]
    src, dst = net.blobs['data'], net.blobs[end]
    src.reshape(1,3,h,w)
    src.data[0] = preprocess(net, guide)
    net.forward(end=end)
    guide_features = dst.data[0].copy()

    def objective_guide(dst):
        x = dst.data[0].copy()
        y = guide_features
        ch = x.shape[0]
        x = x.reshape(ch,-1)
        y = y.reshape(ch,-1)
        A = x.T.dot(y) # compute the matrix of dot-products with guide features
        dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best

    _=deepdream(net, img, end=end, objective=objective_guide)


if __name__ == '__main__':
    model_path = '/home/consti/Work/software/src/caffe/models/bvlc_googlenet/'
    net_fn   = model_path + 'deploy.prototxt'
    param_fn = model_path + 'bvlc_googlenet.caffemodel'

    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    open('tmp.prototxt', 'w').write(str(model))

    net = caffe.Classifier(
            'tmp.prototxt',
            param_fn,
            mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
            channel_swap = (2,1,0) # the reference model has channels in BGR order instead of RGB
            )

    img = np.float32(PIL.Image.open('sky1024px.jpg'))

    deepdream_multi_frame(net, img)

