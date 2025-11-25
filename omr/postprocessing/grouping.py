from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from scipy.spatial import KDTree

from omr.models.detected_symbol import DetectedSymbol
from omr.models.symbols import Symbol


@dataclass
class SymbolGroup:
    """Intermediate representation of grouped raw symbols."""

    group_type: str  # 'note', 'rest', or 'barline'
    main_symbol: Union[DetectedSymbol, None] = None  # The notehead, rest, or barline
    stem: Optional[DetectedSymbol] = None
    flag: Optional[DetectedSymbol] = None
    accidental: Optional[DetectedSymbol] = None
    dots: List[DetectedSymbol] = field(default_factory=list)
    x_pos: float = 0.0

    def __post_init__(self):
        if self.main_symbol:
            self.x_pos = self.main_symbol.bbox.x_center


def group_symbols(symbols: List[DetectedSymbol]) -> List[SymbolGroup]:
    """
    Groups individual detected symbols (heads, stems, flags, dots) into logical units.
    """
    # segregate symbols by type
    noteheads = [s for s in symbols if s.symbol_class in Symbol.get_noteheads()]
    stems = [s for s in symbols if s.symbol_class == Symbol.STEM]
    flags = [s for s in symbols if s.symbol_class in Symbol.get_flags()]
    rests = [s for s in symbols if s.symbol_class in Symbol.get_rests()]
    accidentals = [s for s in symbols if s.symbol_class in Symbol.get_accidentals()]
    dots = [s for s in symbols if s.symbol_class == Symbol.AUGMENTATION_DOT]
    barlines = [s for s in symbols if s.symbol_class == Symbol.BAR_LINE]

    # build spatial trees
    stem_tree = _build_tree(stems)
    flag_tree = _build_tree(flags)
    accidental_tree = _build_tree(accidentals)
    dot_tree = _build_tree(dots)

    used_indices = set()
    groups: List[SymbolGroup] = []

    # process noteheads
    for nh in noteheads:
        group = SymbolGroup(group_type="note", main_symbol=nh)

        # associate stem
        if stem_tree:
            dist, idx = stem_tree.query((nh.bbox.x_center, nh.bbox.y_center))
            if dist < nh.bbox.width * 2:
                group.stem = stems[idx]
                used_indices.add(("stem", idx))

                # associate flag (attached to stem)
                if flag_tree:
                    dist_f, idx_f = flag_tree.query(
                        (stems[idx].bbox.x_center, stems[idx].bbox.y_top)
                    )
                    if dist_f < stems[idx].bbox.height * 1.5:
                        group.flag = flags[idx_f]
                        used_indices.add(("flag", idx_f))

        # associate accidental (lft of notehead)
        if accidental_tree:
            indices = accidental_tree.query_ball_point(
                (nh.bbox.x_center - nh.bbox.width, nh.bbox.y_center),
                r=nh.bbox.width * 2,
            )
            for idx in indices:
                if accidentals[idx].bbox.x_center < nh.bbox.x_center:
                    group.accidental = accidentals[idx]
                    used_indices.add(("accidental", idx))
                    break

        # associate dots (right of notehead)
        if dot_tree:
            indices = dot_tree.query_ball_point(
                (nh.bbox.x_center + nh.bbox.width, nh.bbox.y_center),
                r=nh.bbox.width * 2,
            )
            for idx in indices:
                if dots[idx].bbox.x_center > nh.bbox.x_center:
                    group.dots.append(dots[idx])
                    used_indices.add(("dot", idx))

        groups.append(group)

    # process rests
    for r in rests:
        group = SymbolGroup(group_type="rest", main_symbol=r)

        if dot_tree:
            indices = dot_tree.query_ball_point(
                (r.bbox.x_center + r.bbox.width, r.bbox.y_center), r=r.bbox.width * 2
            )
            for idx in indices:
                if ("dot", idx) not in used_indices:
                    group.dots.append(dots[idx])
                    used_indices.add(("dot", idx))
        groups.append(group)

    # barlines
    for b in barlines:
        groups.append(SymbolGroup(group_type="barline", main_symbol=b))

    # sort by x position
    return sorted(groups, key=lambda x: x.x_pos)


def _build_tree(symbols: List[DetectedSymbol]) -> Optional[KDTree]:
    """Helper to safely build a KDTree from symbol centers."""
    if not symbols:
        return None
    points = [(s.bbox.x_center, s.bbox.y_center) for s in symbols]
    return KDTree(points)
