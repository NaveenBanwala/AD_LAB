import math
from turtle import Screen, Turtle


def heart_x(t):
    return 16 * math.sin(t) ** 3


def heart_y(t):
    return 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)


def blend(c1, c2, t):
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


screen = Screen()
screen.title("Heart Visualization")
screen.bgcolor("#0b0b12")
screen.setup(width=900, height=700)
screen.colormode(255)
screen.tracer(0)

t = Turtle(visible=False)
t.speed(0)
t.pensize(2)

scale = 18
steps = 1200
start = (255, 40, 80)
end = (255, 140, 200)

# Smooth heart outline with gradient
t.penup()
for i in range(steps + 1):
    angle = (2 * math.pi) * i / steps
    x = heart_x(angle) * scale
    y = heart_y(angle) * scale
    color = blend(start, end, i / steps)
    if i == 0:
        t.goto(x, y)
        t.pendown()
    t.pencolor(color)
    t.goto(x, y)

# Fill the heart with a soft glow (scaled inner layers)
for layer in range(12):
    factor = 1 - (layer + 1) * 0.06
    glow = blend((255, 70, 110), (120, 30, 60), layer / 11)
    t.penup()
    for i in range(steps + 1):
        angle = (2 * math.pi) * i / steps
        x = heart_x(angle) * scale * factor
        y = heart_y(angle) * scale * factor
        if i == 0:
            t.goto(x, y)
            t.pendown()
            t.fillcolor(glow)
            t.begin_fill()
        t.goto(x, y)
    t.end_fill()

# Write name inside heart
t.penup()
t.goto(0, -20)
t.pencolor("white")
t.write("Naveen", align="center", font=("Georgia", 28, "bold"))

screen.update()
screen.mainloop()
