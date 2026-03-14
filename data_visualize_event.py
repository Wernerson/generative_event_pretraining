import argparse
import cv2
import glob
import os
from tqdm import tqdm

def show_video(args):
    def overlay_images(img1, img2, threshold=254):
        """
        Overlay non-white areas of img2 onto img1.

        Args:
            img1 (np.ndarray): Background image.
            img2 (np.ndarray): Overlay image (same size as img1).
            threshold (int): Pixel intensity threshold to detect non-white areas.

        Returns:
            np.ndarray: Combined image.
        """
        if img1.shape != img2.shape:
            raise ValueError("Image sizes do not match.")

        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        img2_fg = cv2.bitwise_and(img2, mask_3ch)
        img1_bg = cv2.bitwise_and(img1, cv2.bitwise_not(mask_3ch))

        result = cv2.add(img1_bg, img2_fg)
        return result
    
    cap = cv2.VideoCapture(0)
    output_width = 640
    output_height = 480
    fps = 20
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(args.load_dir, 'vis.mp4'), fourcc, fps, (output_width, output_height))

    event_names = glob.glob(os.path.join(args.load_dir, "event_image", "*.png"))
    event_names = sorted(event_names, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))

    image_names = sorted(glob.glob(os.path.join(args.load_dir, "warpped", "*.png")))
    image_names = sorted(image_names, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
    num_event_images = len(event_names)
    j = 0
    for i in tqdm(range(num_event_images)):
        event_image = cv2.imread(event_names[i])
        image = cv2.imread(image_names[j])
        final = overlay_images(image, event_image)
        out.write(final)
        if args.vis:
            cv2.imshow("event_image", final)
            cv2.waitKey(1)

        event_timestamp = event_names[i].split("/")[-1].split(".")[0]
        image_timestamp = image_names[j].split("/")[-1].split(".")[0]
        if event_timestamp == image_timestamp:
            j += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", default = "/data/storage/jianwen/DSEC/train_images/zurich_city_00_a/images/left", type = str)
    parser.add_argument("--frame_rate", default = 20, type = int)
    parser.add_argument("--vis", action='store_true', help="If true, show the video in a window. It will be faster if not.")
    args = parser.parse_args()

    print(f"video will be saved in: {os.path.join(args.load_dir, 'vis.mp4')}")
    show_video(args)    