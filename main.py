# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf

import ctypes
import ctypes.util
from ctypes import pointer
import io

import numpy as np
import PIL.Image
import matplotlib.pylab as pl

from lucid.misc.gl.glcontext import create_opengl_context
import OpenGL.GL as gl

from lucid.misc.gl import meshutil
from lucid.misc.gl import glrenderer
import lucid.misc.io as lucid_io
from lucid.misc.tfutil import create_session

from lucid.modelzoo import vision_models
from lucid.optvis import objectives
from lucid.optvis import param
from lucid.optvis.style import StyleLoss, mean_l1_loss
from lucid.optvis.param.spatial import sample_bilinear

from display_results import show_textured_mesh


def main():
    model = create_model()
    create_opengl_context()
    mesh, original_texture, style, texture_size = load_data('article_models/skull.obj', 'article_models/skull.jpg',
                                                       'styles/yellow_newspapper.jpg', 2048)

    neural_net = NeuralNet(model, mesh, original_texture, style, texture_size)

    t_texture, loss_log = neural_net.run(600)

    pl.plot(loss_log)
    pl.legend(['Content Loss', 'Style Loss'])
    pl.show()

    texture = t_texture.eval()
    show_textured_mesh("optimized.html", mesh, texture)
    show_textured_mesh("original.html", mesh, original_texture)


def prepare_image(fn, size=None):
    data = lucid_io.reading.read(fn)
    im = PIL.Image.open(io.BytesIO(data)).convert('RGB')
    if size:
        im = im.resize(size, PIL.Image.ANTIALIAS)
    return np.float32(im)/255.0


def load_data(object_path, texture_path, style_path, texture_size=1024):
    mesh = meshutil.load_obj(object_path)
    mesh = meshutil.normalize_mesh(mesh)
    original_texture = prepare_image(texture_path, (texture_size, texture_size))
    style = prepare_image(style_path)
    return mesh, original_texture, style, texture_size


def create_model():
    model = vision_models.InceptionV1()
    model.load_graphdef()
    return model


class NeuralNet:
    def __init__(self, model, mesh, original_texture, style, texture_size):
        self.model = model
        self.mesh = mesh
        self.original_texture = original_texture
        self.style = style
        self.texture_size = texture_size

    def run(self, step_n=400):
        # create renderer
        renderer = glrenderer.MeshRenderer((1024, 1024))

        # neural net parameters
        googlenet_style_layers = [
            'conv2d2',
            'mixed3a',
            'mixed3b',
            'mixed4a',
            'mixed4b',
            'mixed4c',
        ]

        googlenet_content_layer = 'mixed3b'
        content_weight = 100.0
        style_decay = 0.95

        # create tf new session
        sess = create_session(timeout_sec=0)

        # feeding processed optimizing texture to neural net
        t_texture, t_fragments, t_input, content_var = self.__define_texture_params()
        self.model.import_graph(t_input)

        # style loss
        style_layers = [sess.graph.get_tensor_by_name('import/%s:0' % s)[0] for s in googlenet_style_layers]
        sl = StyleLoss(style_layers, style_decay, loss_func=mean_l1_loss)

        # content loss
        content_layer = sess.graph.get_tensor_by_name('import/%s:0' % googlenet_content_layer)
        content_loss = mean_l1_loss(content_layer[0], content_layer[1]) * content_weight

        # setup optimization
        total_loss = content_loss + sl.style_loss
        t_lr = tf.constant(0.05)
        trainer = tf.train.AdamOptimizer(t_lr)
        train_op = trainer.minimize(total_loss)

        init_op = tf.global_variables_initializer()
        init_op.run()

        content_var.load(self.original_texture)
        sl.set_style({t_input: self.style[None, ...]})
        loss_log = []

        for i in range(step_n):
            fragments = renderer.render_mesh(
                modelview=meshutil.sample_view(10.0, 12.0),
                position=self.mesh['position'], uv=self.mesh['uv'],
                face=self.mesh['face'])
            _, loss = sess.run([train_op, [content_loss, sl.style_loss]], {t_fragments: fragments})
            loss_log.append(loss)
            if i == 0 or (i + 1) % 10 == 0:
                print(len(loss_log), loss)

        return t_texture, loss_log

    def __define_texture_params(self):
        # optimizing texture
        t_texture = param.image(self.texture_size, fft=True, decorrelate=True)[0]

        # original texture - processed
        content_var = tf.Variable(tf.zeros([self.texture_size, self.texture_size, 3]), trainable=False)

        # t_fragments [U, V, _, Alpha]
        # t_uv - coordinates
        # t_alpha - 0 if background
        t_fragments = tf.placeholder(tf.float32, [None, None, 4])
        t_uv = t_fragments[..., :2]
        t_alpha = t_fragments[..., 3:]

        # optimizing texture (input_data) - processed
        t_joined_texture = tf.concat([t_texture, content_var], -1)
        t_joined_frame = sample_bilinear(t_joined_texture, t_uv) * t_alpha
        t_frame_current, t_frame_content = t_joined_frame[..., :3], t_joined_frame[..., 3:]
        t_joined_frame = tf.stack([t_frame_current, t_frame_content], 0)
        t_input = tf.placeholder_with_default(t_joined_frame, [None, None, None, 3])

        return t_texture, t_fragments, t_input, content_var


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
