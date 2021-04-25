use ordered_float::OrderedFloat;
use rand::Rng;

use crate::board::{Board, Coord, Player};
use crate::bot_game::Bot;
use crate::mcts::heuristic::{Heuristic, ZeroHeuristic};
use std::num::NonZeroU32;

pub mod heuristic;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct IdxRange {
    pub start: NonZeroU32,
    pub length: u8,
}

impl IdxRange {
    pub fn iter(&self) -> std::ops::Range<u32> {
        self.start.get()..(self.start.get() + self.length as u32)
    }
}

impl IntoIterator for IdxRange {
    type Item = u32;
    type IntoIter = std::ops::Range<u32>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct Node {
    pub coord: Coord,
    pub parent: Option<u32>,
    pub children: Option<IdxRange>,
    pub wins: u32,
    pub draws: u32,
    pub visits: u32,
}

impl Node {
    fn new(coord: Coord, parent: Option<u32>) -> Self {
        Node {
            coord,
            parent,
            children: None,
            wins: 0,
            draws: 0,
            visits: 0,
        }
    }

    pub fn uct(&self, parent_visits: u32, heuristic: f32) -> f32 {
        let wins = self.wins as f32;
        let draws = self.draws as f32;
        let visits = self.visits as f32;

        //TODO is this really the best heuristic formula? maybe let the heuristic decide the weight as well?
        (wins + 0.5 * draws) / visits +
            (2.0 * (parent_visits as f32).ln() / visits).sqrt() +
            (heuristic / (visits + 1.0))
    }

    /// The estimated value of this node in the range -1..1
    pub fn signed_value(&self) -> f32 {
        (2.0 * (self.wins as f32) + (self.draws as f32)) / (self.visits as f32) - 1.0
    }
}

#[derive(Debug)]
pub struct Evaluation {
    pub best_move: Option<Coord>,
    pub value: f32,
}

pub fn mcts_build_tree<H: Heuristic, R: Rng>(board: &Board, iterations: u32, heuristic: &H, rand: &mut R) -> Vec<Node> {
    let mut tree: Vec<Node> = Vec::new();

    //the actual coord doesn't matter, just pick something
    tree.push(Node::new(Coord::from_o(0), None));

    for _ in 0..iterations {
        let mut curr_node: u32 = 0;
        let mut curr_board = board.clone();

        while !curr_board.is_done() {
            //Init children
            let children = match tree[curr_node as usize].children {
                Some(children) => children,
                None => {
                    let start = tree.len() as u32;
                    tree.extend(curr_board.available_moves().map(|c| Node::new(c, Some(curr_node))));
                    let end = tree.len() as u32;

                    debug_assert!(start != 0 && end != 0 );
                    let children = unsafe {
                        //SAFETY: start and end are not zero because the root node occupies index 0
                        IdxRange {
                            start:NonZeroU32::new_unchecked(start),
                            length: (end - start) as u8,
                        }
                    };
                    tree[curr_node as usize].children = Some(children);
                    children
                }
            };

            //Exploration
            let unexplored_children = children.iter()
                .filter(|&c| tree[c as usize].visits == 0);
            let count = unexplored_children.clone().count();

            if count != 0 {
                let child = unexplored_children.clone().nth(rand.gen_range(0, count))
                    .expect("we specifically selected the index based on the count already");

                curr_node = child;
                curr_board.play(tree[curr_node as usize].coord);

                break;
            }

            //Selection
            let parent_visits = tree[curr_node as usize].visits;

            let selected = children.iter().max_by_key(|&child| {
                let heuristic = heuristic.evaluate(&curr_board);
                let uct = tree[child as usize].uct(parent_visits, heuristic);
                OrderedFloat(uct)
            }).expect("Board is not done, this node should have a child");

            curr_node = selected;
            curr_board.play(tree[curr_node as usize].coord);
        }

        //Simulate
        let curr_player = curr_board.next_player;

        let won_by = loop {
            if let Some(won_by) = curr_board.won_by {
                break won_by;
            }

            curr_board.play(curr_board.random_available_move(rand)
                .expect("No winner, so board is not done yet"));
        };

        //Update
        let mut won = if won_by != Player::Neutral {
            won_by == curr_player
        } else {
            rand.gen()
        };

        loop {
            won = !won;

            let node = &mut tree[curr_node as usize];
            node.visits += 1;
            if won {
                node.wins += 1;
            }

            if let Some(parent) = node.parent {
                curr_node = parent;
            } else {
                break;
            }
        }
    }

    tree
}

pub fn mcts_evaluate<H: Heuristic, R: Rng>(board: &Board, iterations: u32, heuristic: &H, rand: &mut R) -> Evaluation {
    let tree = mcts_build_tree(board, iterations, heuristic, rand);

    let best_move = match tree[0].children {
        None => board.random_available_move(rand),
        Some(children) => {
            children.iter().rev().max_by_key(|&child| {
                tree[child as usize].visits
            }).map(|child| {
                tree[child as usize].coord
            })
        }
    };

    let value = tree[0].signed_value();
    Evaluation { best_move, value }
}

pub struct MCTSBot<H: Heuristic, R: Rng> {
    iterations: u32,
    heuristic: H,
    batch_eval: bool,
    rand: R,
}

impl<R: Rng> MCTSBot<ZeroHeuristic, R> {
    pub fn new(iterations: u32, rand: R) -> MCTSBot<ZeroHeuristic, R> {
        MCTSBot { iterations, heuristic: ZeroHeuristic, batch_eval: false, rand }
    }

    pub fn new_with_batch_eval(iterations: u32, rand: R) -> MCTSBot<ZeroHeuristic, R> {
        MCTSBot { iterations, heuristic: ZeroHeuristic, batch_eval: true, rand }
    }
}

impl<H: Heuristic, R: Rng> MCTSBot<H, R> {
    pub fn new_with_heuristic(iterations: u32, rand: R, heuristic: H) -> MCTSBot<H, R> {
        MCTSBot { iterations, heuristic, batch_eval: false, rand }
    }
}

impl<H: Heuristic, R: Rng> Bot for MCTSBot<H, R> {
    fn play(&mut self, board: &Board) -> Option<Coord> {
        mcts_evaluate(board, self.iterations, &self.heuristic, &mut self.rand).best_move
    }
}
