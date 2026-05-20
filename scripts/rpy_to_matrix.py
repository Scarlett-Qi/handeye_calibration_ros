#!/usr/bin/env python3
# coding: utf-8

import argparse
import math


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def mmul(a, b):
    out = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]
    return out


def main():
    p = argparse.ArgumentParser(description="RPY -> Rotation Matrix")
    p.add_argument("--rpy", required=True, help="rx,ry,rz (deg), e.g. 10,-88,165")
    p.add_argument("--order", choices=["zyx", "xyz"], default="zyx", help="composition order")
    p.add_argument("--name", default="R_cam2gripper", help="yaml key name")
    args = p.parse_args()

    parts = [x.strip() for x in args.rpy.split(",")]
    if len(parts) != 3:
        raise ValueError("--rpy format error, use rx,ry,rz")
    rx_deg, ry_deg, rz_deg = map(float, parts)
    rx_rad = math.radians(rx_deg)
    ry_rad = math.radians(ry_deg)
    rz_rad = math.radians(rz_deg)

    Rx = rx(rx_rad)
    Ry = ry(ry_rad)
    Rz = rz(rz_rad)

    if args.order == "zyx":
        # R = Rz * Ry * Rx
        R = mmul(mmul(Rz, Ry), Rx)
    else:
        # R = Rx * Ry * Rz
        R = mmul(mmul(Rx, Ry), Rz)

    print(f"{args.name}:")
    for row in R:
        print("- - {:.6f}".format(row[0]))
        print("  - {:.6f}".format(row[1]))
        print("  - {:.6f}".format(row[2]))


if __name__ == "__main__":
    main()
