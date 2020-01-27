import numpy as np
import io
from PIL import Image

# def fig2image(fig):
#     fig.canvas.draw()
#     return np.array(fig.canvas.renderer.buffer_rgba())

def fig2image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(io.BytesIO(buf.getvalue()))