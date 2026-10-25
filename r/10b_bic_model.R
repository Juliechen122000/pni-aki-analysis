## =====================================================================
## 10b_bic_model.R  ——  A2：纯 BIC 选模的对照分析
## =====================================================================
## 目的：回答"不使用 AKI 结局挑选展示模型时，主结论是否改变？"
##
## 选模规则（唯一依据，不看任何结局信息）：
##   在全部 19 个已收敛的网格模型中，取 BIC 最小者。
##   若 BIC 最小者即为当前展示解，如实说明；若不是，按实际结果报告，
##   不为保留原模型名而强行选择。
##
## 参照类的选择同样不看结局：取人数最多的类别。
## （原展示解用"AKI 最高类"作参照，那是结局导向的；本处刻意改用 size-based）
##
## 输出：
##   bic_selection_log.csv    选模过程与模型质量
##   bic_model_m1m4.csv       新解的 M1–M4
##   bic_vs_main.csv          与原展示解的逐项对比
## =====================================================================

.libPaths(c(Sys.getenv("RLIB", "./Rlib"), .libPaths()))
WORK <- Sys.getenv("INSPIRE_WORK", "."); OUT <- file.path(WORK, "output")
SD_PNI <- 6.4452

hr <- function(t) cat("\n", strrep("=", 72), "\n", t, "\n", strrep("=", 72), "\n", sep = "")

## ---------- 1. 找全局 BIC 最小 ----------
files <- list.files(OUT, pattern = "^lcmm_grid_onodera_(linear|quad|cubic|ns2|ns3)\\.csv$",
                    full.names = TRUE)
grid <- do.call(rbind, lapply(files, read.csv, stringsAsFactors = FALSE))
grid <- grid[grid$conv %in% c(TRUE, "TRUE") & !is.na(grid$BIC), ]

hr("全部收敛模型的 BIC 排序（仅拟合指标，未使用任何结局信息）")
ord <- grid[order(grid$BIC), c("form", "ng", "BIC", "loglik", "entropy",
                               "minClass", "cls_size")]
print(head(ord, 8), row.names = FALSE)

best <- ord[1, ]
cat(sprintf("\n>>> 纯 BIC 最优：%s ng=%d，BIC=%.1f\n", best$form, best$ng, best$BIC))
cat(sprintf(">>> 当前展示解：quad ng=3，BIC=%.1f\n",
            grid$BIC[grid$form == "quad" & grid$ng == 3]))
if (best$form == "quad" && best$ng == 4) {
  cat(">>> 结论：纯 BIC 最优恰好是 quad 家族的 ng=4，与展示解同函数形式、不同类别数。\n")
} else {
  cat(">>> 注意：纯 BIC 最优并非 quad ng=4，按实际结果报告。\n")
}

write.csv(ord, file.path(OUT, "bic_selection_log.csv"), row.names = FALSE)

## ---------- 2. 载入该解并作诊断 ----------
rds <- file.path(OUT, sprintf("lcm_onodera_%s_ng%d.rds", best$form, best$ng))
if (!file.exists(rds)) stop("缺少 RDS: ", rds)
m <- readRDS(rds)
pp <- m$pprob
P <- as.matrix(pp[, grep("^prob", names(pp)), drop = FALSE])
K <- ncol(P)
pm <- pm <- NULL
PM <- P * log(P); PM[!is.finite(PM)] <- 0
E <- 1 - sum(-PM) / (nrow(P) * log(K))
avepp <- sapply(seq_len(K), function(k) mean(P[pp$class == k, k], na.rm = TRUE))
cs <- 100 * table(factor(pp$class, levels = seq_len(K))) / nrow(pp)

hr(sprintf("模型质量：%s ng=%d", best$form, best$ng))
cat(sprintf("BIC=%.1f  loglik=%.1f  n=%d  K=%d\n", m$BIC, m$loglik, nrow(pp), K))
cat(sprintf("entropy=%.3f  最小/中位 AvePP=%.3f / %.3f  pct70=%.1f%%\n",
            E, min(avepp), median(avepp),
            100 * mean(apply(P, 1, max) >= 0.70)))
cat("类别比例: ", paste(sprintf("C%d=%.1f%%", seq_len(K), cs), collapse = "  "), "\n")

## ---------- 3. 合并结局 ----------
L <- read.csv(file.path(OUT, "lcmm_long_onodera.csv"), stringsAsFactors = FALSE)
names(L) <- sub("^\ufeff", "", names(L))
L <- L[order(L$subject_id, L$time), ]
fst <- L[!duplicated(L$subject_id), c("subject_id", "pni")]
names(fst)[2] <- "pni_post_first"

A <- read.csv(file.path(OUT, "p0_aki_inspire_harmonized.csv"), stringsAsFactors = FALSE)
names(A) <- sub("^\ufeff", "", names(A))
A$open_ <- as.integer(substr(as.character(A$pcs), 5, 5) == "0")
A$emale <- as.integer(A$sex == "M")
A$aki <- A$aki_unified
A$pni_pre <- A$pni_onodera
A <- merge(A, fst, by = "subject_id", all.x = TRUE)

d <- data.frame(subject_id = pp$subject_id, cl = pp$class)
D <- merge(A, d, by = "subject_id")
D <- D[!is.na(D$aki), ]

## 参照类：人数最多的类（不看结局）
refk <- as.integer(names(which.max(table(D$cl))))
D$cl <- relevel(factor(D$cl), ref = as.character(refk))
cat(sprintf("\n参照类（人数最多，size-based，不涉及结局）: C%d\n", refk))

keep <- c("aki", "cl", "pni_pre", "pni_post_first", "age", "emale", "asa",
          "op_duration", "cr", "emop", "open_")
for (v in c("age", "asa", "op_duration", "cr")) D[[v]] <- as.numeric(D[[v]])
Dc <- D[complete.cases(D[, keep]), ]
cat(sprintf("complete-case n=%d  事件=%d\n", nrow(Dc), sum(Dc$aki)))

hr("各类别的大致 AKI 发生率（仅描述，未参与选模）")
print(round(100 * tapply(Dc$aki, Dc$cl, mean), 2))

## ---------- 4. M1–M4 ----------
ors <- function(g, terms) {
  s <- summary(g)$coefficients
  out <- character(length(terms))
  for (i in seq_along(terms)) {
    t <- terms[i]
    if (!t %in% rownames(s)) { out[i] <- NA_character_; next }
    b <- s[t, 1]; se <- s[t, 2]
    out[i] <- sprintf("%.2f (%.2f-%.2f) p=%.3g", exp(b), exp(b - 1.96 * se),
                      exp(b + 1.96 * se), s[t, 4])
  }
  out
}

clt <- grep("^cl", names(coef(glm(aki ~ cl, data = Dc, family = binomial))), value = TRUE)
f <- list(
  M1 = aki ~ cl,
  M2 = aki ~ cl + I(pni_pre / SD_PNI),
  M3 = aki ~ cl + I(pni_pre / SD_PNI) + age + emale + asa + op_duration + cr + emop + open_,
  M4 = aki ~ cl + I(pni_post_first / SD_PNI) + age + emale + asa + op_duration + cr + emop + open_
)

RES <- list()
for (nm in names(f)) {
  g <- glm(f[[nm]], data = Dc, family = binomial)
  tt <- clt
  if (nm %in% c("M2", "M3")) tt <- c(clt, "I(pni_pre/SD_PNI)")
  if (nm == "M4") tt <- c(clt, "I(pni_post_first/SD_PNI)")
  v <- ors(g, tt)
  for (j in seq_along(tt)) {
    RES[[length(RES) + 1]] <- data.frame(
      model = nm, n = nrow(Dc), events = sum(Dc$aki),
      term = tt[j], est = v[j], stringsAsFactors = FALSE)
  }
}

R <- do.call(rbind, RES)
write.csv(R, file.path(OUT, "bic_model_m1m4.csv"), row.names = FALSE)

hr("纯 BIC 选模下的递进模型")
for (nm in names(f)) {
  z <- R[R$model == nm, ]
  cat(sprintf("%s  n=%d ev=%d\n", nm, z$n[1], z$events[1]))
  for (i in seq_len(nrow(z))) cat(sprintf("     %-22s %s\n", z$term[i], z$est[i]))
}

## ---------- 5. 与原展示解对比 ----------
orig <- read.csv(file.path(OUT, "lcmm_adjusted_ors.csv"), stringsAsFactors = FALSE)
orig <- orig[orig$form == "quad" & orig$ng == 3, ]

hr("与原展示解（quad ng=3）的对比")
cat("原展示解参照类 = AKI 最高类；本对照参照类 = 人数最多类。\n")
cat("两者的 OR 锚点不同，故类别 OR 数值不可直接相减；\n")
cat("可比的是：①类别效应是否在 M4 归零；②PNI 项自身的估计。\n\n")
cat("--- 原解（quad ng=3）PNI 项 ---\n")
print(orig[grepl("PNI", orig$term), c("model", "term", "OR_CI")], row.names = FALSE)
cat("\n--- 新解 PNI 项 ---\n")
print(R[grepl("PNI", R$term), c("model", "term", "est")], row.names = FALSE)

cmp <- data.frame(
  item = c("model", "BIC", "entropy", "minClass(%)", "n_analysis", "events",
           "ref_class_rule", "M4_class_effect"),
  original = c("quad ng=3", "118288.1", "0.416", "20.4", "5950", "432",
               "AKI-highest class", "0.77 (0.49-1.21) / 0.81 (0.62-1.04)"),
  bic_based = c(sprintf("%s ng=%d", best$form, best$ng),
                sprintf("%.1f", m$BIC), sprintf("%.3f", E),
                sprintf("%.1f", min(cs)), as.character(nrow(Dc)),
                as.character(sum(Dc$aki)), "largest class (size-based)",
                paste(sprintf("%s", R$est[R$model == "M4" & R$term %in% clt]),
                      collapse = " / ")),
  stringsAsFactors = FALSE)
write.csv(cmp, file.path(OUT, "bic_vs_main.csv"), row.names = FALSE)
print(cmp, row.names = FALSE)

cat("\n已写出 bic_selection_log.csv / bic_model_m1m4.csv / bic_vs_main.csv\n")
