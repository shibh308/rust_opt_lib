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
        let target = (if self.maximize {
            "maximize "
        } else {
            "minimize "
        })
        .to_string()
            + &target_str;
        f.debug_struct("LPProblem")
            .field("target", &target)
            .field("constraints", &constraints)
            .field("non_positive", &self.non_positive)
            .finish()
    }
}

impl LPProblem {
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
    fn solve_by_simplex(&self) -> (f64, Vec<f64>) {
        (0.0, Vec::new())
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
        /*
        maximize
                 2x + 3y
        s.t.
                  x +  y <= 6
                  x + 3y <= 12
                 2x +  y <= 10
         */
        let x = 0;
        let y = 1;
        let mut prob = LPProblem::new(2, true, vec![(2., x), (3., y)], vec![true; 2]);
        prob.add_constraint(vec![(1., x), (1., y)], LessEq, 6.);
        prob.add_constraint(vec![(1., x), (3., y)], LessEq, 12.);
        prob.add_constraint(vec![(2., x), (1., y)], LessEq, 10.);
        let (value, var) = prob.solve_by_simplex();
        println!("max value: {},  var: {:?}", value, var);
    }
}
