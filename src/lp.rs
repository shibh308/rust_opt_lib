use crate::matrix::Matrix;
use std::fmt::Formatter;

#[derive(PartialEq, Copy, Clone)]
enum CompareOp {
    Eq,
    LessEq,
    GreaterEq,
}

impl CompareOp {
    fn to_string(&self) -> &str {
        match self {
            CompareOp::Eq => "==",
            CompareOp::GreaterEq => ">=",
            CompareOp::LessEq => "<=",
        }
    }
}

#[derive(Clone)]
struct LPProblem {
    n: usize,
    maximize: bool,
    target: Vec<(f64, usize)>,
    // ax <= b
    constraints: Vec<(Vec<(f64, usize)>, CompareOp, f64)>,
    non_positive: Vec<bool>,
}

impl std::fmt::Debug for LPProblem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (target, constraints) = self.get_output_format();
        f.debug_struct("LPProblem")
            .field("target", &target)
            .field("constraints", &constraints)
            .field("non_positive", &self.non_positive)
            .field("maximize", &self.maximize)
            .finish()
    }
}

impl std::fmt::Display for LPProblem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let (target, constraints) = self.get_output_format();
        let target_type = (if self.maximize {
            "maximize"
        } else {
            "minimize"
        })
        .to_string();
        let _ = write!(f, "{}:\n\t{}\n", target_type, target);
        let _ = write!(
            f,
            "subject to:\n{}",
            constraints
                .iter()
                .fold("".to_string(), |s, x| s + &format!("\t{}\n", x).to_string())
        );
        write!(
            f,
            "{}",
            self.non_positive
                .iter()
                .enumerate()
                .filter_map(|(i, b)| if *b {
                    Some(format!("\tx_{} >= 0\n", i))
                } else {
                    None
                })
                .fold("".to_string(), |s, x| s + &x)
        )
    }
}

impl LPProblem {
    const EPS: f64 = 1e-6;
    fn new(n: usize, maximize: bool, target: Vec<(f64, usize)>, non_positive: Vec<bool>) -> Self {
        assert_eq!(non_positive.len(), n);
        assert_ne!(n, 0);
        let mut before_idx = target[0].1;
        for (i, (_, idx)) in target.iter().enumerate() {
            assert!(i == 0 || before_idx < *idx);
            assert!(*idx < n);
            before_idx = *idx;
        }
        // TODO: idx strictly increasing
        LPProblem {
            n,
            maximize,
            target,
            constraints: Vec::new(),
            non_positive,
        }
    }
    fn add_constraint(&mut self, ax: Vec<(f64, usize)>, op: CompareOp, b: f64) {
        let mut before_idx = ax[0].1;
        for (i, (_, idx)) in ax.iter().enumerate() {
            assert!(i == 0 || before_idx < *idx);
            assert!(*idx < self.n);
            before_idx = *idx;
        }
        // TODO: idx strictly increasing
        self.constraints.push((ax, op, b));
    }
    fn is_valid(&self, x: &Vec<f64>) -> bool {
        assert_eq!(self.n, x.len());
        self.constraints.iter().fold(0, |ok_cnt, (ax, op, b)| {
            let sum = ax
                .iter()
                .fold(0.0, |sum, (rate, idx)| sum + *rate * x[*idx]);
            if match op {
                CompareOp::Eq => (sum - b).abs() < Self::EPS,
                CompareOp::GreaterEq => sum > *b || (sum - b).abs() < Self::EPS,
                CompareOp::LessEq => sum < *b || (sum - b).abs() < Self::EPS,
            } {
                ok_cnt + 1
            } else {
                ok_cnt
            }
        }) == self.constraints.len()
    }
    // 標準形への変形
    // minimizeに変更した時の-1倍や変数の変更は考慮しない (TODO)
    fn standard_form(&self) -> LPProblem {
        // minimizeのmaximizeへの変更
        let target: Vec<_> = self
            .target
            .iter()
            .map(|(val, idx)| (val * if self.maximize { 1.0 } else { -1.0 }, *idx))
            .collect();

        // 非負制約がない変数の調整
        let mut constraints = Vec::new();
        let mut nex_n = self.n;
        let mut non_pos_ids = vec![0; self.n];
        for i in 0..self.n {
            if !self.non_positive[i] {
                non_pos_ids[i] = nex_n;
                nex_n += 1;
            }
        }
        for (ax, op, b) in &self.constraints {
            let mut nex_ax = Vec::new();
            for (rate, idx) in ax {
                nex_ax.push((*rate, *idx));
                if !self.non_positive[*idx] {
                    nex_ax.push((-1.0 * *rate, non_pos_ids[*idx]));
                }
            }
            constraints.push((nex_ax, op.clone(), b.clone()));
        }

        let mut new_constraints = Vec::new();
        // 制約式の置き換え
        for (ax, op, b) in &constraints {
            let (pos, neg) = match op {
                CompareOp::LessEq => (true, false),
                CompareOp::GreaterEq => (false, true),
                CompareOp::Eq => (true, true),
            };
            if pos {
                new_constraints.push((ax.clone(), CompareOp::LessEq, b.clone()));
            }
            if neg {
                let ax: Vec<_> = ax
                    .iter()
                    .clone()
                    .map(|(val, idx)| (val * -1.0, *idx))
                    .collect();
                new_constraints.push((ax.clone(), CompareOp::LessEq, -1.0 * b));
            }
        }
        let mut std_prob = LPProblem::new(nex_n, true, target, vec![true; nex_n]);
        std_prob.constraints = new_constraints;
        std_prob
    }
    fn is_standard(&self) -> bool {
        if !self.maximize || self.non_positive.iter().fold(0, |i, x| i + *x as usize) != self.n {
            false
        } else {
            for (_, op, _) in &self.constraints {
                if let CompareOp::Eq | CompareOp::GreaterEq = op {
                    return false;
                }
            }
            true
        }
    }
    fn get_output_format(&self) -> (String, Vec<String>) {
        let target_str = self
            .target
            .iter()
            .map(|(rate, idx)| format!("({} * x_{})", rate, idx))
            .fold("".to_string(), |s, t| format!("{} + {}", s, t))[3..]
            .to_string();
        let constraints: Vec<_> = self
            .constraints
            .iter()
            .map(|(v, op, b)| {
                let ax = v
                    .iter()
                    .map(|(rate, idx)| format!("({} * x_{})", rate, idx))
                    .fold("".to_string(), |s, t| format!("{} + {}", s, t))[3..]
                    .to_string();
                format!("{} {} {}", ax, op.to_string(), b)
            })
            .collect();
        (target_str, constraints)
    }
    fn add_slack_variable(&mut self) {
        assert!(self.is_standard());
        let m = self.constraints.len();
        let mut constraints = Vec::new();
        for (i, (ax, op, b)) in self.constraints.iter().enumerate() {
            let mut ax = ax.clone();
            assert!(op == &CompareOp::LessEq);
            ax.push((1.0, self.n + i));
            constraints.push((ax, CompareOp::Eq, *b));
        }
        self.n += m;
        self.constraints = constraints;
        self.non_positive = vec![true; self.n];
    }
    fn solve_by_simplex(self) -> (LPProblem, f64, Vec<f64>) {
        let mut slack_prob = self.clone().standard_form();
        slack_prob.add_slack_variable();
        let (val, x) = slack_prob._solve_by_simplex(self.n);
        (slack_prob, val, x)
    }
    fn make_dict(&self, nonbasic_idxes: &Vec<usize>) -> Vec<(usize, Vec<(f64, usize)>, f64)> {
        let n = self.n - nonbasic_idxes.len();
        let mut is_nonbasic = vec![false; self.n];
        for idx in nonbasic_idxes {
            is_nonbasic[*idx] = true;
        }
        let mut dict = vec![None; self.n];
        for (ax, op, b) in &self.constraints {
            assert!(op == &CompareOp::Eq);
            let mut basic_idx = None;
            let mut dict_elm = Vec::new();
            let mut dict_per = 0.0;
            for (per, idx) in ax {
                if is_nonbasic[*idx] {
                    dict_elm.push((*per, *idx));
                } else {
                    assert!(basic_idx.is_none());
                    dict_per = *per;
                    basic_idx = Some(idx);
                }
            }

            let dict_elm: Vec<_> = dict_elm
                .iter()
                .map(|(per, idx)| (-1.0 * per / dict_per, *idx))
                .collect();
            let dict_right = b / dict_per;

            assert!(basic_idx.is_some());
            let basic_idx = *basic_idx.unwrap();
            dict[basic_idx] = Some((basic_idx, dict_elm, dict_right));
        }
        let dict: Vec<_> = dict.iter().filter_map(|x| x.clone()).collect();
        assert_eq!(dict.len(), n);
        dict
    }
    fn rebuild_dict_target(
        &mut self,
        dict: &Vec<(usize, Vec<(f64, usize)>, f64)>,
        idxes_data: &Vec<(bool, usize)>,
        remove_idx: usize,
        add_idx: usize,
    ) -> (Vec<(usize, Vec<(f64, usize)>, f64)>, f64) {
        assert!(idxes_data[add_idx].0);
        assert!(!idxes_data[remove_idx].0);
        // x[add_idx] = ax + b
        // 右辺の中からremove_idxに対応する部分を持ってきて、
        let dict_target = dict[idxes_data[add_idx].1].clone();
        // x_3 = (1/2)x_0 - 3x_1 + 12
        // x_0 = 2x_3 + 6x_1 - 24
        let per = dict_target
            .1
            .iter()
            .find_map(|(per, id)| if *id == remove_idx { Some(per) } else { None });
        // TODO: Noneの時の処理 (どうすればいい？)
        let per = per.unwrap();
        let new_dict_target_1: Vec<_> = dict_target
            .1
            .iter()
            .map(|(now_per, id)| {
                if *id == remove_idx {
                    (1.0 / per, add_idx)
                } else {
                    (now_per / (-1.0 * per), *id)
                }
            })
            .collect();
        let new_dict_target = (remove_idx, new_dict_target_1, dict_target.2 / (-1.0 * per));
        let mut new_dict = Vec::new();
        for (idx, ax, b) in dict {
            if *idx == add_idx {
                new_dict.push(new_dict_target.clone());
            } else {
                match ax
                    .iter()
                    .find_map(|(per, id)| if *id == remove_idx { Some(per) } else { None })
                {
                    Some(per) => {
                        let mut ax: Vec<_> = ax
                            .iter()
                            .filter(|(_, id)| *id != remove_idx)
                            .cloned()
                            .collect();
                        for (per1, idx) in &new_dict_target.1 {
                            ax.push((*per * *per1, *idx));
                        }
                        new_dict.push((*idx, ax, *b + *per * new_dict_target.2));
                    }
                    None => {
                        new_dict.push((*idx, ax.clone(), *b));
                    }
                }
            }
        }

        let mut new_target = Vec::new();
        let mut target_per = 0.0;
        for (per, id) in &self.target {
            if *id == remove_idx {
                target_per = *per;
                for (per1, idx) in &new_dict_target.1 {
                    new_target.push((*per * *per1, *idx));
                }
            } else {
                new_target.push((*per, *id));
            }
        }

        /*
        println!("old: {:?}", dict_target);
        println!("new: {:?}", new_dict_target);
        println!("old: {:?}", dict);
        println!("new: {:?}", new_dict);
         */

        // 正規の形に戻す
        let bucket_recalc = |v: Vec<(f64, usize)>, self_n: usize| -> Vec<_> {
            let mut bucket = vec![0.0; self_n];
            for (a, x) in v {
                bucket[x] += a;
            }
            bucket
                .iter()
                .enumerate()
                .filter_map(|(i, per)| {
                    if per.abs() < Self::EPS {
                        None
                    } else {
                        Some((*per, i))
                    }
                })
                .collect()
        };

        self.target = bucket_recalc(new_target, self.n);
        new_dict = new_dict
            .iter()
            .map(|(i, ax, b)| (*i, bucket_recalc(ax.clone(), self.n), *b))
            .collect();
        new_dict.sort_by(|x, y| (x.0).cmp(&y.0));
        (new_dict, new_dict_target.2 * target_per)
    }
    fn calc_basic_values(&self, dict: &Vec<(usize, Vec<(f64, usize)>, f64)>, x: &mut Vec<f64>) {
        for (i, ax, b) in dict {
            let mut now_b = *b;
            for (per, idx) in ax {
                now_b += per * x[*idx];
            }
            x[*i] = now_b
        }
        assert!(self.is_valid(&x));
    }
    fn coef_mat(&self) -> Matrix<f64> {
        let m = self.constraints.len();
        let mut mat = Matrix::new(m, self.n, 0.0f64);
        for (i, (ax, _, _)) in self.constraints.iter().enumerate() {
            for (rate, j) in ax {
                *mat.get_mut(i, *j) = *rate;
            }
        }
        mat
    }
    fn get_idxes_data(&self, nonbasic_idxes: &Vec<usize>) -> Vec<(bool, usize)> {
        let mut idx_data = vec![(true, 0usize); self.n];
        let mut is_basic = vec![true; self.n];
        for i in nonbasic_idxes {
            is_basic[*i] = false;
        }
        let mut basic_prog = 0;
        let mut nonbasic_prog = 0;
        for i in 0..self.n {
            if is_basic[i] {
                idx_data[i] = (true, basic_prog);
                basic_prog += 1;
            } else {
                idx_data[i] = (false, nonbasic_prog);
                nonbasic_prog += 1;
            }
        }
        idx_data
    }

    fn check_regularity(coef_mat: &Matrix<f64>, nonbasic_idxes: &Vec<usize>) -> bool {
        let mut mat = Matrix::new(coef_mat.h, nonbasic_idxes.len(), 0.0f64);
        for (j1, j2) in nonbasic_idxes.iter().enumerate() {
            for i in 0..coef_mat.h {
                *mat.get_mut(i, j1) = coef_mat.get(i, *j2);
            }
        }
        mat.gaussian_elimination().0 == nonbasic_idxes.len()
    }

    fn _solve_by_simplex(&mut self, n: usize) -> (f64, Vec<f64>) {
        let mut nonbasic_idxes = (0..n).collect();
        let mut x = vec![0.0f64; self.n];
        let mut offset_val = 0.0;

        // 係数行列を用意しておく
        let coef_mat = self.coef_mat();

        // 基底変数 = 非基底変数の線型結合 + 定数    の形の辞書を作る
        let mut dict = self.make_dict(&nonbasic_idxes);

        // 辞書から基底変数の値を求める
        self.calc_basic_values(&dict, &mut x);

        // 変数が一次独立でない場合は弾く
        assert!(
            Self::check_regularity(&coef_mat, &nonbasic_idxes),
            "Variables must have linear independence"
        );
        // 原点が実行領域にない場合は弾く
        // TODO: 原点が実行可能領域でない時にも動くようにする
        assert!(x.iter().find(|val| val.is_sign_negative()).is_none());

        loop {
            let idxes_data = self.get_idxes_data(&nonbasic_idxes);
            self.calc_basic_values(&dict, &mut x);

            assert!(Self::check_regularity(&coef_mat, &nonbasic_idxes));
            assert!(x.iter().find(|val| val.is_sign_negative()).is_none());

            /*
            println!();
            println!("{:?}", self);
            println!("{:?}", nonbasic_idxes);
            println!("{:?}", dict);
            println!("{:?}", x);
            println!(
                "{:?}",
                self.target
                    .iter()
                    .fold(offset_val, |sum, (a, i)| sum + *a * x[*i])
            );
             */

            // 最小添字規則で, 被約費用 (目的関数の係数) が正である変数を探して動かす
            let change_pair = match self.target.iter().fold(None, |x, y| {
                if x.is_none() && Self::EPS < y.0 {
                    Some(y.1)
                } else {
                    x
                }
            }) {
                Some(max_idx) => {
                    let mut max_mov = f64::INFINITY;
                    for (_, ax, b) in &dict {
                        let ax = ax.iter().find(|x| x.1 == max_idx);
                        if ax.is_some() {
                            let per = ax.unwrap().0;
                            if per < -Self::EPS {
                                max_mov = max_mov.min(b / (-1.0 * per));
                            }
                        }
                    }
                    if max_mov.is_infinite() {
                        // 無限に大きくできるので, その時の値を
                        x[max_idx] = f64::INFINITY;
                        for (i, ax, b) in &dict {
                            let mut now_b = *b;
                            for (per, idx) in ax {
                                now_b += per * x[*idx];
                            }
                            x[*i] = now_b
                        }
                        return (f64::INFINITY, x);
                    }
                    Some((max_idx, max_mov))
                }
                None => None,
            };
            if change_pair.is_none() {
                let z = self
                    .target
                    .iter()
                    .fold(offset_val, |sum, (a, i)| sum + *a * x[*i]);
                // println!("end\n");
                return (z, x);
            }
            let (max_idx, max_mov) = change_pair.unwrap();
            x[max_idx] = max_mov;

            // 非基底変数の値を元に基底変数を計算し直す
            self.calc_basic_values(&dict, &mut x);

            // 基底変数を変更する
            let origin_idxes: Vec<_> = (0..self.n)
                .filter(|i| !idxes_data[*i].0 && *i != max_idx)
                .collect();
            assert_eq!(origin_idxes.len(), n - 1);
            let cand_iter = (0..self.n).filter(|i| idxes_data[*i].0 && x[*i].abs() < Self::EPS);
            let swap_idx = cand_iter.clone().find_map(|cand| {
                let mut idxes = origin_idxes.clone();
                idxes.push(cand);
                // 一次独立であるような物を探してくる
                if Self::check_regularity(&coef_mat, &idxes) {
                    Some(cand)
                } else {
                    None
                }
            });
            assert!(swap_idx.is_some());
            let swap_idx = swap_idx.unwrap();
            nonbasic_idxes = (0..self.n)
                .filter(|i| *i == swap_idx || (!idxes_data[*i].0 && *i != max_idx))
                .collect();

            // targetとdictの更新
            let (new_dict, add_offset) =
                self.rebuild_dict_target(&dict, &idxes_data, max_idx, swap_idx);
            dict = new_dict;
            offset_val += add_offset;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::lp::{CompareOp::*, LPProblem};

    #[test]
    fn lp_standard() {
        let x1 = 0;
        let x2 = 1;
        let x3 = 2;
        let mut prob = LPProblem::new(
            3,
            false,
            vec![(3., x1), (4., x2), (-2., x3)],
            vec![true, true, false],
        );
        prob.add_constraint(vec![(2., x1), (1., x2)], Eq, 4.);
        prob.add_constraint(vec![(1., x1), (-2., x3)], LessEq, 8.);
        prob.add_constraint(vec![(3., x2), (1., x3)], GreaterEq, 6.);

        assert!(!prob.is_standard());
        let std_prob = prob.standard_form();
        assert!(std_prob.is_standard());
    }

    #[test]
    fn lp_solve() {
        let x = 0;
        let y = 1;

        let mut prob = LPProblem::new(2, true, vec![(1., x), (2., y)], vec![true; 2]);
        prob.add_constraint(vec![(1., x), (1., y)], LessEq, 6.);
        prob.add_constraint(vec![(1., x), (3., y)], LessEq, 12.);
        prob.add_constraint(vec![(2., x), (1., y)], LessEq, 10.);
        println!("------original problem------");
        println!("{}", prob);
        let (slack_prob, value, var) = prob.solve_by_simplex();
        println!("------converted problem------");
        println!("{}", slack_prob);
        println!("max value: {},  var: {:?}", value, var);
        println!();
        println!();

        let mut prob = LPProblem::new(2, true, vec![(2., x), (1., y)], vec![true; 2]);
        prob.add_constraint(vec![(1., x), (-2., y)], LessEq, 4.);
        prob.add_constraint(vec![(-1., x), (1., y)], LessEq, 2.);
        println!("------original problem------");
        println!("{}", prob);
        let (slack_prob, value, var) = prob.solve_by_simplex();
        println!("------converted problem------");
        println!("{}", slack_prob);
        println!("max value: {},  var: {:?}", value, var);
        println!();
        println!();
    }
}
