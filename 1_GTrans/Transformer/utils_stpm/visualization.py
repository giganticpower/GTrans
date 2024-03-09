import os
import numpy as np
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib
import matplotlib.pyplot as plt

def img_norm(scores):
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    vmax = vmax * 0.5 + vmin * 0.5
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    return norm

def plt_fig(test_img, scores, img_scores, f_vis, gts, threshold, cls_threshold, save_dir, class_name):
    num = len(scores)
    norm = img_norm(scores)
    f_vis = np.asarray(f_vis)
    f1, f2, f3 = f_vis[:, 0], f_vis[:, 1], f_vis[:, 2]
    f4, f5, f6 = f_vis[:, 3], f_vis[:, 4], f_vis[:, 5]
    norm1, norm2, norm3 = img_norm(f1), img_norm(f2), img_norm(f3)
    norm4, norm5, norm6 =img_norm(f4), img_norm(f5), img_norm(f6)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    vmax = vmax * 0.5 + vmin * 0.5
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        # fig_img, ax_img = plt.subplots(1, 4, figsize=(9, 3), gridspec_kw={'width_ratios': [4, 4, 4, 3]})
        fig_img, ax_img = plt.subplots(3, 3)
        fig_img.subplots_adjust(wspace=0.05, hspace=0)
        for ax_i in ax_img:
            for ax_ij in ax_i:
                ax_ij.axes.xaxis.set_visible(False)
                ax_ij.axes.yaxis.set_visible(False)

        ax_img[0, 0].imshow(img) # 原图
        ax_img[0, 0].title.set_text('Input image')
        ax_img[0, 1].imshow(gt, cmap='gray')  # 标签
        ax_img[0, 1].title.set_text('GroundTruth')
        ax_img[0, 2].imshow(heat_map, cmap='jet', norm=norm, interpolation='none')  # 分割热力图
        ax_img[0, 2].imshow(vis_img, cmap='gray', alpha=0.7, interpolation='none')
        ax_img[0, 2].title.set_text('Segmentation')

        '''
        black_mask = np.zeros((int(mask.shape[0]), int(3 * mask.shape[1] / 4)))
        ax_img[3].imshow(black_mask, cmap='gray')
        ax = plt.gca()
        if img_scores[i] > cls_threshold:
            cls_result = 'nok'
        else:
            cls_result = 'ok'

        ax.text(0.05,
                0.89,
                'Detected anomalies',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.79,
                '------------------------',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.72,
                'Results',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.67,
                '------------------------',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.59,
                '\'{}\''.format(cls_result),
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='r',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.47,
                'Anomaly scores: {:.2f}'.format(img_scores[i]),
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.37,
                '------------------------',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.30,
                'Thresholds',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.25,
                '------------------------',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.17,
                'Classification: {:.2f}'.format(cls_threshold),
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.07,
                'Segementation: {:.2f}'.format(threshold),
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax_img[3].title.set_text('Classification')
        '''

        ax_img[1, 0].imshow(f1[i] * 255, cmap='jet', norm=norm1, interpolation='none')
        ax_img[1, 0].imshow(img, cmap='gray', alpha=0.7, interpolation='none')
        # ax_img[1, 0].title.set_text('Map0_56×56')

        ax_img[1, 1].imshow(f2[i] * 255, cmap='jet', norm=norm2, interpolation='none')
        ax_img[1, 1].imshow(img, cmap='gray', alpha=0.7, interpolation='none')
        # ax_img[1, 1].title.set_text('Map1_28×28')

        ax_img[1, 2].imshow(f3[i] * 255, cmap='jet', norm=norm3, interpolation='none')
        ax_img[1, 2].imshow(img, cmap='gray', alpha=0.7, interpolation='none')
        # ax_img[1, 2].title.set_text('Map2_14×14')


        ax_img[2, 0].imshow(f4[i] * 255, cmap='jet', norm=norm4, interpolation='none')
        ax_img[2, 0].imshow(img, cmap='gray', alpha=0.7, interpolation='none')
        # ax_img[2, 0].title.set_text('Map 01')

        ax_img[2, 1].imshow(f5[i] * 255, cmap='jet', norm=norm5, interpolation='none')
        ax_img[2, 1].imshow(img, cmap='gray', alpha=0.7, interpolation='none')
        # ax_img[2, 1].title.set_text('Map 02')

        ax_img[2, 2].imshow(f6[i] * 255, cmap='jet', norm=norm6, interpolation='none')
        ax_img[2, 2].imshow(img, cmap='gray', alpha=0.7, interpolation='none')
        # ax_img[2, 2].title.set_text('Map 12')



        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=300, bbox_inches='tight')
        plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x