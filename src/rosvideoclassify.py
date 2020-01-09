from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import cv2
import copy
import glob
import argparse
import facenet
import align.detect_face
from timeit import default_timer as timer

# rospackage
import rospy
from std_msgs.msg import String

minsize = 20  # minimum size of face
fd_threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709  # scale factor
input_image_size = 160
fr_threshold = 0.8 # 認識のしきい値
rospyrate = 2 # publishのHz

# initialize rosnode
class ROSFaceClassification:
    detected = False
    pub = None

    def __init__(self, rate = rospyrate):
        rospy.init_node('detect_registered_person')
        self.pub = rospy.Publisher('/faceclassifier/string', String, queue_size=1)
        self.rate = rospy.Rate(rate)

    def publish(self, boolmsg):
        self.pub.publish(boolmsg)

def main(args):
    margin = args.margin
    obj = ROSFaceClassification()

    with tf.Graph().as_default():
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        gpu_options = tf.GPUOptions(allow_growth=True) # GPUのメモリ割り当て方法を変更
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                    log_device_placement=False))
        with sess.as_default():
        # 顔検出のネットワーク作成　MTCNN
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None) 

            image_paths = glob.glob(args.reg_paths) # 登録済み画像のフォルダ
            nrof_images = len(image_paths) #登録済み画像の数(only one person)

            # 登録済み画像から顔のみを抽出したリストを作成
            images = load_and_align_data(image_paths, nrof_images, pnet, rnet, onet, args)
            nrof_images = len(images) #登録に成功した顔の数(only one person)

            # Load the model
            facenet.load_model(args.model)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb_reg = sess.run(embeddings, feed_dict=feed_dict)  # 登録済み画像の特徴ベクトル抽出

            # カメラ映像/動画ファイルの取得
            video_capture = cv2.VideoCapture(0) # camera input
            video_capture.set(cv2.CAP_PROP_FPS, 3)
            print('Start Recognition')

            #fps計算 初期化
            frame_num = 1
            accum_time = 0
            curr_fps = 0
            prev_time = timer()
            fps = "FPS: ??"

            while not rospy.is_shutdown():
                ret, frame = video_capture.read()
                if ret == False:
                    break

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                #frame = cv2.resize(frame, (640, 352)) # 入力画像をリサイズ
                bounding_boxes, _ = align.detect_face.detect_face(frame, minsize,
                                                pnet, rnet, onet, fd_threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                # print(bounding_boxes)
                print('Detected_FaceNum: %d' % nrof_faces, end='')

                if nrof_faces > 0:  #顔を検出した場合
                    det = bounding_boxes[:, 0:4]
                    frame_size = np.asarray(frame.shape)[0:2]
                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    v_bb = np.zeros((nrof_faces,4), dtype=np.int32)

                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))
                        v_bb[i][0] = np.maximum(det[i][0]-margin/2, 0)   # 左上 x(横)
                        v_bb[i][1] = np.maximum(det[i][1]-margin/2, 0)   # 左上 y(縦)
                        v_bb[i][2] = np.minimum(det[i][2]+margin/2, frame_size[1])   # 右下 x(横)
                        v_bb[i][3] = np.minimum(det[i][3]+margin/2, frame_size[0])   # 右下 y(縦)
                        cropped.append(frame[v_bb[i][1]:v_bb[i][3], v_bb[i][0]:v_bb[i][2], :])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(misc.imresize(cropped[i], 
                                    (input_image_size, input_image_size), interp='bilinear'))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                            interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))

                        cv2.rectangle(frame, (v_bb[i][0], v_bb[i][1]), (v_bb[i][2], v_bb[i][3]), (0, 255, 0), 2)

                        feed_dict = {images_placeholder: scaled_reshape[i],
                                                            phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict) # 特徴ベクトルの抽出

                        # 識別(登録済み画像の特徴ベクトルとのユークリッド距離を計算)
                        dist_ave = cal_distance(emb_reg, emb_array, nrof_images)
                        print('  %1.4f  ' % dist_ave, end='')

                        if dist_ave < fr_threshold: # 認識のしきい値
                            #plot result idx under box
                            text_x = v_bb[i][0]
                            text_y = v_bb[i][3] + 20
                            print('Find registered person', end='')
                            obj.publish('owner')
                            cv2.rectangle(frame, (v_bb[i][0], v_bb[i][1]), 
                                                        (v_bb[i][2], v_bb[i][3]), (0, 0, 255), 2)
                        else:
                            print('', end='')
                            obj.publish('face')
                else:  #顔非検出の場合
                    print('  Alignment Failure', end='')
                    obj.publish('none')
                print('')
                obj.rate.sleep()

                #frame_num表示
                cv2.putText(frame, str(frame_num), (3,30), cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.50, (255, 0, 0), thickness=2)
                frame_num += 1

                #fps計算
                curr_time = timer()
                exec_time = curr_time - prev_time
                prev_time = curr_time
                accum_time = accum_time + exec_time
                curr_fps = curr_fps + 1
                if accum_time > 1:
                    accum_time = accum_time - 1
                    fps = "FPS: " + str(curr_fps)
                    curr_fps = 0
                cv2.putText(frame, fps, (3,15), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0), thickness=2)

                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # video_capture.release()
            cv2.destroyAllWindows()

def load_and_align_data(image_paths, nrof_images,pnet, rnet, onet, args):
    img_list = []
    for image in image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB') # 画像読み込み RGB形式

        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, fd_threshold, factor) # 顔検出
        if len(bounding_boxes) < 1: # 顔が検出されなかった場合
            print("can't detect face", image)
            continue
        det = np.squeeze(bounding_boxes[0,0:4]) #顔の検出ポイント
        cropped = cropped_face(det, img, img_size, args)
        img_list.append(cropped)
    if nrof_images > 1:
        images = np.stack(img_list) # 登録済み画像から顔のみ抽出したリスト
    else:
        images = img_list
    return images

def cropped_face(det, img, img_size, args):
    margin = args.margin
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)   # 左上 x(横)
    bb[1] = np.maximum(det[1]-margin/2, 0)   # 左上 y(縦)
    bb[2] = np.minimum(det[2]+margin/2, img_size[1])   # 右下 x(横)
    bb[3] = np.minimum(det[3]+margin/2, img_size[0])   # 右下 y(縦)

    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:] # bounding boxの場所指定
    aligned = misc.imresize(cropped, (input_image_size, input_image_size), interp='bilinear') # クロッピングしてリサイズ
    aligned = facenet.prewhiten(aligned)
    return aligned

def cal_distance(emb_reg, emb_video, nrof_images):
    dist = np.zeros(nrof_images, dtype=np.float64)
    dist_ave = 0.
    cnt = 1
    for j in range(nrof_images):
        dist[j] = np.sqrt(np.sum(np.square(np.subtract(emb_reg[j,:], emb_video[0, :])))) #ユークリッド距離計算
    dist.sort()   #距離が短い順に並び替え
    for x in range(3):  # kNN, k=3
        dist_ave += dist[x]
        cnt += 1
        if cnt > len(dist):
            break

    dist_ave = dist_ave / float(cnt-1) # 登録済み画像とのユークリッド距離(最近点3個)
    return dist_ave


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    # parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('reg_paths', type=str, help='The path of registered human faces')
    parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
    help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float, help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))