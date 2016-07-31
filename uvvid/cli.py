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

        coord = uvvid.find_cursor(frame, template_frame)

        if prev_frame is not None:
            drawing, color = uvvid.is_cursor_drawing(frame, prev_frame, coord)
            if debug:
                top_left = (coord[0] - template_frame.shape[0]//2,
                            coord[1] - template_frame.shape[1]//2)
                bottom_right = (coord[0] + template_frame.shape[0]//2,
                                coord[1] + template_frame.shape[1]//2)
                cv.rectangle(frame, top_left, bottom_right, (255, 255, 255), 1)
                ds = "Drawing" if drawing else "Idle"
                cv.rectangle(frame, (frame.shape[1] - 60, 20),
                                    (frame.shape[1] - 40, 40), color, -1)
                cv.putText(frame, ds, (frame.shape[1] - 150, 40),
                           cv.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255), 1)

        cv.imshow('frame', frame)
        prev_frame = frame
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
