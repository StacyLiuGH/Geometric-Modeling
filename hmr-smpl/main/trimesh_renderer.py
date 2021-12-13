import cv2
import numpy as np
import trimesh
import trimesh.transformations as trans

from vis_util import load_faces


class TrimeshRenderer(object):

    def __init__(self, img_size=(224, 224), focal_length=10.):
        self.h, self.w = img_size[0], img_size[1]
        self.focal_length = focal_length
        self.faces = load_faces()

    def __call__(self, verts, img=None, img_size=None, bg_color=None):

        if img is not None:
            h, w = img.shape[:2]
        elif img_size is not None:
            h, w = img_size[0], img_size[1]
        else:
            h, w = self.h, self.w

        mesh = self.mesh(verts)
        scene = mesh.scene()

        if bg_color is not None:
            bg_color = np.zeros(4)

        image_bytes = scene.save_image(resolution=(w, h), background=bg_color, visible=True)
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            x1, x2 = 0, img.shape[1]
            y1, y2 = 0, img.shape[0]

            alpha_mesh = image[:, :, 3] / 255.0
            alpha_image = 1.0 - alpha_mesh

            for c in range(0, 3):
                img[y1:y2, x1:x2, c] = (alpha_mesh * image[:, :, c] + alpha_image * img[y1:y2, x1:x2, c])

            image = img

        return image

    def mesh(self, verts):
        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces,
                               vertex_colors=[125, 125, 125, 255],
                               face_colors=[0, 0, 0, 0],
                               use_embree=False,
                               process=False)

        # apply transform: z axis is other way around in trimesh
        transform = trans.rotation_matrix(np.deg2rad(-180), [1, 0, 0], mesh.centroid)
        mesh.apply_transform(transform)
        
        return mesh
 
    def show_mesh_scene(self, mesh):
        scene = mesh.scene()
        scene.show()
 	
    def save_obj(self, mesh):
        filename = r'../smpl_output/smpl.obj'
        with open(filename, 'w', encoding='utf-8') as f:
            mesh.export(f, file_type='obj', include_normals=False, include_texture=False)
