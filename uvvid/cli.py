import click

import cv2 as cv

from .core import UVVID

DEBUG = False


@click.group()
@click.option('--debug/--no-debug', default=True)
@click.pass_context
def cli(ctx, debug):
    ctx.obj = {}
    ctx.obj['DEBUG'] = debug


@cli.command()
@click.option('--cursor', required=True, type=click.Path(exists=True))
@click.option('--video', required=True, type=click.Path(exists=True))
@click.option('--drawing-window', type=int, default=10)
@click.option('--drawing-ratio', type=float, default=0.2)
@click.pass_context
def view(ctx, cursor, video, drawing_window, drawing_ratio):
    debug = ctx.obj['DEBUG']

    uvvid = UVVID(drawing_window=drawing_window, drawing_ratio=drawing_ratio)
    template_frame = cv.imread(cursor, 0)
    cap = cv.VideoCapture(video)
    prev_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break

        if prev_frame is not None:
            curr_pos = int(cap.get(cv.cv.CV_CAP_PROP_POS_MSEC))
            uvvid.generate_strokes(frame, prev_frame, template_frame, curr_pos)

        cv.imshow('frame', frame)
        prev_frame = frame
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    if debug:
        uvvid.generate_json("input.mp4", "cursor_1.png", "interpolation",
                            "compressed_xyz.mp3", "background.png", (10, 11),
                            "15", "75")
    cv.waitKey(0)
