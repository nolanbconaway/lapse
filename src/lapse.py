import datetime
import multiprocessing
from contextlib import contextmanager
from pathlib import Path
from typing import List

import cv2
import imageio
import numpy as np
from tqdm.auto import tqdm


def extract_date(p: Path) -> datetime.datetime:
    """My images have isoformatted filenames."""
    return datetime.datetime.fromisoformat(p.name[:-4])


def date_stamp(path: Path, _: np.ndarray) -> str:
    """Date annotator helper."""
    return extract_date(path).date().isoformat()


def date_hour_stamp(path: Path, _: np.ndarray) -> str:
    """Date-hour annotator helper."""
    return extract_date(path).strftime("%Y-%m-%d %H")


def load_image(
    p: Path,
    process_func: callable = None,
    cast_to_rbg: bool = True,
) -> np.ndarray:
    """Load a single image from a path, optionally applying transformations.

    args:
        p: Path to the image to load.
        process_func: Function to apply to the image post-load.
        cast_to_rbg: Whether to convert opencv's BGR to the more common RGB order.
            Default: True.
    """
    p = cv2.imread(str(p.resolve()))
    if cast_to_rbg:
        p = p[:, :, ::-1]

    if process_func is None:
        process_func = lambda x: x

    return process_func(p)


def pass_loader_args(arg):
    """For use in multiprocessing when I need to pass kwargs.

    Just zip up the kwargs and pass em to this func.
    """
    x, kw = arg
    return load_image(x, **kw)


def multi_load_image(paths: List[Path], **kwargs) -> List[np.ndarray]:
    """Load multiple images via multiprocessing, optionally with an aggregator.

    It is best to use a process func here which aggregates the image somehow, as this will
    load ALL images as arrays into memory which can quickly generate a RAM problem for
    large datasets.

    Accepts the same kwargs as load_image. but with an additional procs argument which
    specifies the number of processes to use.

    Also shows a tqdm progress bar.
    """

    if "procs" in kwargs:
        procs = kwargs.pop("procs")
    else:
        procs = multiprocessing.cpu_count() - 1

    N = len(paths)
    res = []
    with multiprocessing.Pool(procs) as pool, tqdm(total=N) as pbar:
        for img in pool.imap(pass_loader_args, [(x, kwargs) for x in paths]):
            res.append(img)
            pbar.update()
    return res


@contextmanager
def VideoWriter(*args, **kwargs):
    """A context manager for a video writer."""
    cap = cv2.VideoWriter(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def apply_annotation(text: str, image: np.ndarray) -> np.ndarray:
    """Apply text to an image.

    Args:
        text: Text to apply to the image.
        image: Image to apply the text to.

    Returns:
        The image with the text applied.
    """
    height, width, _ = image.shape
    image = cv2.putText(
        img=np.array(image),  # just in case the image is a view
        text=text,
        org=(int(width * 0.01), int(height * 0.03)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 0),
        thickness=4,
    )
    image = cv2.putText(
        img=image,
        text=text,
        org=(int(width * 0.01), int(height * 0.03)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 255, 255),
        thickness=1,
    )
    return image


def make_movie(
    image_paths: List[Path],
    save_path: Path,
    fps: int,
    annotate_func: callable = None,
    **loader_kwargs,
) -> None:
    """Make a movie and save it to a path.

    Args:
        image_paths: List of paths to images to make a movie from.
        save_path: Path to save the movie to.
        fps: Frames per second of the movie.
        annotate_func: Function to apply to each image to annotate it. This function
            should have signature ``f(image_path, image_data)`` and return a string that
            will be annotated to the image in the top-left corner. Ordinarily I use this
            to annotate the image with the date it was taken, using the date from the
            filename.

    Additional kwargs are passed to load_image. Has no return value.
    """
    nonempty_paths = [p for p in image_paths if p.stat().st_size > 0]
    height, width, _ = load_image(nonempty_paths[0], **loader_kwargs).shape
    with VideoWriter(
        str(save_path), cv2.VideoWriter_fourcc(*"DIVX"), fps, (width, height)
    ) as video:
        for image_path in tqdm(nonempty_paths):
            image = load_image(image_path, **loader_kwargs)
            if annotate_func:
                image = apply_annotation(annotate_func(image_path, image), image)
            video.write(image)


def make_gif(
    image_paths: list,
    save_path: Path,
    fps: int,
    annotate_func: callable = None,
    procs: int = None,
    **loader_kwargs,
):
    """Make a gif animation and save it to a path.

    Uses multiprocessing to load images in parallel, as gif animations are usually
    on the small side and my computer has 64gb ram.

    Args:
        image_paths: List of paths to images to make a gif from.
        save_path: Path to save the gif to.
        fps: Frames per second (equivalent to 1/duration of frame).
        annotate_func: Function to apply to each image to annotate it. This function
            should have signature ``f(image_path, image_data)`` and return a string that
            will be annotated to the image in the top-left corner. Ordinarily I use this
            to annotate the image with the date it was taken, using the date from the
            filename.
        procs: Number of processes to use. Defaults to the number of cores minus one.

    Additional kwargs are passed to load_image. Has no return value.
    """
    nonempty_paths = [p for p in image_paths if p.stat().st_size > 0]
    images = []
    for image_path, image in zip(
        nonempty_paths, multi_load_image(nonempty_paths, procs=procs, **loader_kwargs)
    ):
        if annotate_func:
            image = apply_annotation(annotate_func(image_path, image), image)
        images.append(image)

    imageio.mimsave(save_path, images, format="GIF", loop=0, duration=1 / fps)
