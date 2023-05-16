from pcl.s3d import S3D

backbone3D = S3D()
backbone3D.load_weights(file_weight='weights/S3D_kinetics400.pt')
