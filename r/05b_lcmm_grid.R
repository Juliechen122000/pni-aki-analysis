## =====================================================================
## 05b_lcmm_grid.R  ——  LCMM 轨迹模型 P0 网格（分块并行）
## =====================================================================
## 用法：
##   Rscript 05b_lcmm_grid.R linear  simple
##   Rscript 05b_lcmm_grid.R quad    onodera
##   共 5 个时间函数 x 2 个 PNI 算法，按需起进程
##
## 与 05a 的区别（为何重写）
##   1. lcmm 2.2.2 的 gridsearch() 在 R 4.6 下必崩：
##        错误 'names' attribute [8] must be the same length as the vector [7]
##      根源在源码第 40 行 `mc$B <- substitute(random(minit), environment())`：
##      当 hlme 调用内不存在 B 形参时，给 call 对象 $<- 新名字会破坏 names 属性。
##      显式传入 B = m0$best 亦无效（已实测）。random() 也未导出，无法绕过。
##      → 改为自实现多起点，行为完全可控。
##   2. 数据源改为 p0_lcmm_build_long.py 产出的 lcmm_long_{simple,onodera}.csv，
##      建在「术前 PNI 入组」的新主分析队列（7,995）内，与正文表 1 自洽。
##      （analysis_dataset.csv 是旧「轨迹合格」队列 4,826，口径不一致，弃用）
##   3. 早停规则按 P0 规格：至少 3 组起始；此后连续 2 组 loglik 差 < tol 才停。
##
## 术语：LCMM（lcmm::hlme, random=~1, nwg=TRUE），非 Nagin 的 GBTM/LCGA
## =====================================================================

.libPaths(c(Sys.getenv("RLIB", "./Rlib"), .libPaths()))
suppressPackageStartupMessages({
  library(lcmm); library(splines)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L)
  stop("usage: Rscript 05b_lcmm_grid.R <linear|quad|cubic|ns2|ns3> [simple|onodera]")
FORM <- args[1]
ALGO <- if (length(args) >= 2L) args[2] else "simple"
if (!ALGO %in% c("simple", "onodera")) stop("algo must be simple or onodera")

WORK <- Sys.getenv("INSPIRE_WORK", ".")
OUT   <- file.path(WORK, "output")
CHUNK <- file.path(OUT, sprintf("lcmm_grid_%s_%s.csv", ALGO, FORM))
LONG  <- file.path(OUT, sprintf("lcmm_long_%s.csv", ALGO))

## ---- 多起点配置 ----
REP_BY_NG     <- c("2" = 8, "3" = 8, "4" = 6, "5" = 6, "6" = 5)
MAXITER_DEPART <- 800      # 随机起点阶段迭代上限（粗筛，求快）
MAXITER_FINAL  <- 5000     # 精算阶段迭代上限
MIN_START      <- 3L       # P0 规格：至少 3 组起始
TOL_LL         <- 1e-3    # 连续 2 组 loglik 差 < tol 视为似然面已探明
PERTURB_SCALE  <- 0.5      # 初值扰动强度（相对各参数量级）

SEED <- 20261023 + match(FORM, c("linear","quad","cubic","ns2","ns3")) * 101 +
        if (ALGO == "onodera") 7 else 0
set.seed(SEED)

hr <- function(t = "") cat("\n", strrep("=", 70), "\n", t,
                           "\n", strrep("=", 70), "\n", sep = "")

cat(sprintf("[%s/%s] 启动  种子=%d  起点数=%s\n", ALGO, FORM, SEED,
            paste(sprintf("ng%s:%d", names(REP_BY_NG), REP_BY_NG), collapse = " ")))

## ---------- 1. 长格式数据 ----------
if (!file.exists(LONG)) stop("缺少长格式数据: ", LONG)
L <- read.csv(LONG, stringsAsFactors = FALSE)
names(L) <- sub("^\ufeff", "", names(L))
if (!all(c("subject_id","time","pni") %in% names(L)))
  stop("长格式缺少必需列")
L <- L[order(L$subject_id, L$time), ]
hr(sprintf("%s/%s —— 数据", ALGO, FORM))
cat("长格式行数:", nrow(L), "  患者数:", length(unique(L$subject_id)), "\n")
print(table(L$time))

## ---------- 2. 时间函数 ----------
FORMS <- list(linear = "time",
              quad   = "poly(time,2,raw=TRUE)",
              cubic  = "poly(time,3,raw=TRUE)",
              ns2    = "ns(time,2)",
              ns3    = "ns(time,3)")
if (!FORM %in% names(FORMS)) stop("未知时间函数: ", FORM)
mx <- as.formula(paste0("pni ~ ", FORMS[[FORM]]))

## ---------- 3. 分类诊断 ----------
diag_cls <- function(f) {
  K <- f$ng
  if (K == 1) return(list(minAvePP = NA, medAvePP = NA, pct70 = NA,
                          entropy = NA, cls_size = NA))
  pp <- f$pprob
  P  <- as.matrix(pp[, grep("^prob", names(pp)), drop = FALSE])
  cl <- pp$class
  avepp <- sapply(seq_len(K), function(k) mean(P[cl == k, k], na.rm = TRUE))
  PM <- P * log(P); PM[!is.finite(PM)] <- 0
  ent <- 1 - sum(-PM) / (nrow(P) * log(K))
  cs  <- 100 * table(factor(cl, levels = seq_len(K))) / sum(table(cl))
  list(minAvePP = round(min(avepp, na.rm = TRUE), 3),
       medAvePP = round(median(avepp, na.rm = TRUE), 3),
       pct70    = round(100 * mean(apply(P, 1, max) >= 0.70), 1),
       entropy  = round(ent, 3),
       cls_size = paste(sprintf("%.1f", cs), collapse = "|"))
}

## ---------- 4. 初值扰动 ----------
perturb <- function(B, scale = PERTURB_SCALE) {
  B + rnorm(length(B), 0, scale * pmax(abs(B), 0.5))
}

## ---------- 5. 多起点拟合（含早停） ----------
fit_multi <- function(ng, p1, rep_n, seed0) {
  ## 第 1 组：自动初值（B = ng=1 模型）
  m <- try(hlme(fixed = mx, mixture = mx, random = ~ 1, subject = "subject_id",
                ng = ng, nwg = TRUE, B = p1, data = L,
                maxiter = MAXITER_FINAL, verbose = FALSE), silent = TRUE)
  if (inherits(m, "try-error") || m$conv != 1) {
    m <- try(hlme(fixed = mx, mixture = mx, random = ~ 1, subject = "subject_id",
                  ng = ng, nwg = TRUE, data = L,
                  maxiter = MAXITER_FINAL, verbose = FALSE), silent = TRUE)
  }
  if (inherits(m, "try-error") || m$conv != 1)
    return(list(m = NULL, nstart = 0L, lls = numeric(0)))

  best  <- m
  Bbest <- m$best
  lls   <- m$loglik
  nconv <- 1L

  for (i in seq_len(rep_n)) {
    set.seed(seed0 + i)
    mi <- try(hlme(fixed = mx, mixture = mx, random = ~ 1, subject = "subject_id",
                   ng = ng, nwg = TRUE, B = perturb(Bbest), data = L,
                   maxiter = MAXITER_DEPART, verbose = FALSE), silent = TRUE)
    if (!inherits(mi, "try-error") && mi$conv == 1) {
      nconv <- nconv + 1L
      lls <- c(lls, mi$loglik)
      if (mi$loglik > best$loglik) { best <- mi; Bbest <- mi$best }
    } else {
      lls <- c(lls, NA_real_)
    }
    ## 早停：至少 MIN_START 组，且最近 3 组里连续 2 个差值都 < tol
    k <- length(lls)
    if (k >= MIN_START && all(!is.na(lls[(k - 2):k]))) {
      d1 <- abs(lls[k] - lls[k - 1]); d2 <- abs(lls[k - 1] - lls[k - 2])
      if (d1 < TOL_LL && d2 < TOL_LL) break
    }
  }

  ## 精算：用搜索到的最优初值跑满迭代
  fin <- try(hlme(fixed = mx, mixture = mx, random = ~ 1, subject = "subject_id",
                  ng = ng, nwg = TRUE, B = Bbest, data = L,
                  maxiter = MAXITER_FINAL, verbose = FALSE), silent = TRUE)
  if (!inherits(fin, "try-error") && fin$conv == 1 && fin$loglik >= best$loglik)
    best <- fin
  list(m = best, nstart = length(lls), nconv = nconv, lls = lls)
}

blank_row <- function(ng, note = "", nstart = 0L) {
  data.frame(algo = ALGO, form = FORM, ng = ng, AIC = NA_real_, BIC = NA_real_,
             loglik = NA_real_, conv = FALSE, niter = NA_integer_,
             nstart = nstart, nstart_conv = 0L,
             minClass = NA_real_, minAvePP = NA_real_, medAvePP = NA_real_,
             pct70 = NA_real_, entropy = NA_real_, cls_size = "",
             note = note, seed = SEED, elapsed_min = NA_real_,
             stringsAsFactors = FALSE)
}

## ---------- 6. 断点续跑 ----------
done <- integer(0); old <- NULL
if (file.exists(CHUNK)) {
  old <- try(read.csv(CHUNK, stringsAsFactors = FALSE), silent = TRUE)
  if (!inherits(old, "try-error") && "ng" %in% names(old)) done <- old$ng
  cat(sprintf("[%s/%s] 续跑：已完成 ng = %s\n", ALGO, FORM,
              if (length(done)) paste(done, collapse = ",") else "(无)"))
}
append_row <- function(r) {
  old <<- if (is.null(old) || nrow(old) == 0) r else rbind(old, r)
  write.csv(old, CHUNK, row.names = FALSE)
}

## ---------- 7. ng = 1 基准 ----------
t_all <- Sys.time()
p1 <- hlme(fixed = mx, random = ~ 1, subject = "subject_id",
           ng = 1, nwg = FALSE, data = L, maxiter = MAXITER_FINAL, verbose = FALSE)
if (!1L %in% done) {
  r <- data.frame(algo = ALGO, form = FORM, ng = 1, AIC = p1$AIC, BIC = p1$BIC,
                  loglik = p1$loglik, conv = p1$conv == 1, niter = p1$niter,
                  nstart = 1L, nstart_conv = 1L, minClass = 100,
                  minAvePP = NA, medAvePP = NA, pct70 = NA, entropy = NA,
                  cls_size = "100.0", note = "ng1 baseline", seed = SEED,
                  elapsed_min = 0, stringsAsFactors = FALSE)
  append_row(r)
  cat(sprintf("[%s/%s] ng=1  BIC=%.2f\n", ALGO, FORM, p1$BIC))
}

## ---------- 8. ng = 2..6 ----------
for (ng in 2:6) {
  if (ng %in% done) { cat(sprintf("[%s/%s] ng=%d 已完成，跳过\n", ALGO, FORM, ng)); next }
  t0 <- Sys.time()
  rep_n <- REP_BY_NG[[as.character(ng)]]
  cat(sprintf("\n[%s/%s] --- ng=%d (最多 %d 组起点) ---\n", ALGO, FORM, ng, rep_n))

  res <- fit_multi(ng, p1, rep_n, SEED + ng * 1000)
  if (is.null(res$m)) {
    cat("      未收敛，放弃该格\n")
    r <- blank_row(ng, "not converged", res$nstart)
    r$elapsed_min <- round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2)
    append_row(r); next
  }
  best <- res$m
  dg <- diag_cls(best)
  cs <- as.numeric(strsplit(dg$cls_size, "\\|")[[1]])

  saveRDS(best, file.path(OUT, sprintf("lcm_%s_%s_ng%d.rds", ALGO, FORM, ng)))
  cat(sprintf("      loglik=%.3f BIC=%.2f conv=%s niter=%d  起点%d组(收敛%d)\n",
              best$loglik, best$BIC, best$conv == 1, best$niter,
              res$nstart, res$nconv))
  cat(sprintf("      minClass=%.1f%% minAvePP=%s entropy=%s pct70=%s\n",
              min(cs), dg$minAvePP, dg$entropy, dg$pct70))

  r <- data.frame(algo = ALGO, form = FORM, ng = ng, AIC = best$AIC, BIC = best$BIC,
                  loglik = best$loglik, conv = best$conv == 1, niter = best$niter,
                  nstart = res$nstart, nstart_conv = res$nconv,
                  minClass = min(cs), minAvePP = dg$minAvePP,
                  medAvePP = dg$medAvePP, pct70 = dg$pct70, entropy = dg$entropy,
                  cls_size = dg$cls_size,
                  note = sprintf("multistart<=%d", rep_n), seed = SEED,
                  elapsed_min = round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2),
                  stringsAsFactors = FALSE)
  append_row(r)
}

hr(sprintf("%s/%s 完成  总耗时 %.1f 分钟", ALGO, FORM,
           as.numeric(difftime(Sys.time(), t_all, units = "mins"))))
print(old[, c("ng","BIC","conv","nstart","minClass","minAvePP","entropy","pct70","elapsed_min")],
      row.names = FALSE)
