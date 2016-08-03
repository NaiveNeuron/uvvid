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
@click.pass_context
def view(ctx, cursor, video):
    debug = ctx.obj['DEBUG']

    uvvid = UVVID()
    template_frame = cv.imread(cursor, 0)
    cap = cv.VideoCapture(video)
    prev_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break

        if prev_frame is not None:
            uvvid.generate_strokes(frame, prev_frame, template_frame)

        cv.imshow('frame', frame)
        prev_frame = frame
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    if debug:
        uvvid.__debug_points__(prev_frame, uvvid.get_strokes())
        cv.imshow('debug points', prev_frame)
        cv.waitKey(0)
