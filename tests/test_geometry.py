from vision_core.geometry import box_area, box_iou, clip_box, expand_box


def test_identical_boxes_iou_is_one() -> None:
    box = (10.0, 20.0, 40.0, 80.0)
    assert box_iou(box, box) == 1.0


def test_disjoint_boxes_iou_is_zero() -> None:
    assert box_iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)) == 0.0


def test_known_half_shift_iou() -> None:
    # Intersection 5x10=50, union=150, IoU=1/3.
    iou = box_iou((0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0))
    assert abs(iou - (1.0 / 3.0)) < 1e-9


def test_box_area_and_clip() -> None:
    assert box_area((0.0, 0.0, 10.0, 5.0)) == 50.0
    clipped = clip_box((-4.0, -2.0, 12.0, 20.0), 10, 10)
    assert clipped[0] == 0.0
    assert clipped[1] == 0.0
    assert clipped[2] == 10.0
    assert clipped[3] == 10.0


def test_expand_box_stays_in_frame() -> None:
    expanded = expand_box((2.0, 2.0, 6.0, 6.0), 8.0, 10, 10)
    assert expanded == (0.0, 0.0, 10.0, 10.0)
