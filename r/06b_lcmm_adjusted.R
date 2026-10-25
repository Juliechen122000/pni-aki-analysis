## =====================================================================
## 06b_lcmm_adjusted.R  ——  候选轨迹解的「增量价值」决定性检验
## =====================================================================
## 问题：轨迹类别的 AKI 梯度（quad ng3: 2.85 / 4.97 / 11.19 %）看起来
##       很大，但它可能只是「术前 PNI 水平」的重新包装。
##       若如此，轨迹分组对连续 PNI 没有增量价值，不能进主分析。
##
## 四个递进模型（以 quad ng3 / ns2 ng3 两个候选解各跑一遍）：
##   M1  aki ~ class                                  粗关联
##   M2  aki ~ class + 术前PNI                        去掉「基线水平」后还剩多少
##   M3  aki ~ class + 术前PNI + Spec A/B 协变量      完全调整
##   M4  aki ~ class + 术后首个PNI + 协变量           去掉「术后即时水平」后还剩多少
##                                                   （M4 若类别效应归零 →
##                                                    斜率/形状无独立价值）
##
## 暴露尺度：PNI 按 −1 SD 报告（SD 冻结 = 6.4452，与主分析一致）
## =====================================================================

.libPaths(c(Sys.getenv("RLIB", "./Rlib"), .libPaths()))
WORK <- Sys.getenv("INSPIRE_WORK", "."); OUT <- file.path(WORK, "output")
SD_PNI <- 6.4452

L <- read.csv(file.path(OUT, "lcmm_long_onodera.csv"), stringsAsFactors = FALSE)
names(L) <- sub("^\ufeff", "", names(L))
AKI <- read.csv(file.path(OUT, "p0_aki_inspire_harmonized.csv"), stringsAsFactors = FALSE)
names(AKI) <- sub("^\ufeff", "", names(AKI))

## 术后首个 PNI（time = 0 时段）
L <- L[order(L$subject_id, L$time), ]
fst <- L[!duplicated(L$subject_id), c("subject_id", "pni")]
names(fst)[2] <- "pni_post_first"

A <- AKI
A$open_ <- as.integer(substr(as.character(A$pcs), 5, 5) == "0")
A <- merge(A, fst, by = "subject_id", all.x = TRUE)
A$aki  <- A$aki_unified
A$sev  <- A$sev_unified
A$pni_pre <- A$pni_onodera
A$emale <- as.integer(A$sex == "M")

ors <- function(g, terms) {
  s <- summary(g)$coefficients
  out <- character(length(terms))
  for (i in seq_along(terms)) {
    t <- terms[i]
    if (!t %in% rownames(s)) { out[i] <- NA_character_; next }
    b <- s[t, 1]; se <- s[t, 2]
    out[i] <- sprintf("%.2f (%.2f-%.2f) p=%.4g", exp(b), exp(b - 1.96 * se),
                      exp(b + 1.96 * se), s[t, 4])
  }
  out
}

RES <- list()
for (cd in list(list(form="quad", ng=3), list(form="ns2", ng=3),
                list(form="quad", ng=4), list(form="quad", ng=2),
                list(form="cubic", ng=2))) {
  fp <- file.path(OUT, sprintf("lcm_onodera_%s_ng%d.rds", cd$form, cd$ng))
  if (!file.exists(fp)) next
  pp <- readRDS(fp)$pprob
  pp <- pp[, c("subject_id", "class")]
  names(pp)[2] <- "cl"
  D <- merge(A, pp, by = "subject_id")
  D <- D[!is.na(D$aki), ]
  K <- length(unique(D$cl))
  refk <- which.max(tapply(D$aki, D$cl, mean) * 1)          # AKI 最高的类作参照
  refk <- as.integer(names(refk)[1])
  D$cl <- relevel(factor(D$cl), ref = as.character(refk))
  keep <- c("aki","cl","pni_pre","pni_post_first","age","emale","asa",
            "op_duration","cr","emop","open_")
  num <- c("age","asa","op_duration","cr")
  for (v in num) D[[v]] <- as.numeric(D[[v]])
  Dc <- D[stats::complete.cases(D[, keep]), ]
  cat(sprintf("\n=== %s ng=%d === 轨迹队列 n=%d  complete-case n=%d (丢弃 %d)\n",
              cd$form, cd$ng, nrow(D), nrow(Dc), nrow(D) - nrow(Dc)))
  cat(sprintf("    参照类 = C%d（AKI 最高）  各类事件数: %s\n", refk,
              paste(sprintf("C%s:%d", names(table(Dc$cl)),
                            as.integer(tapply(Dc$aki, Dc$cl, sum))), collapse=" ")))

  f1 <- aki ~ cl
  f2 <- aki ~ cl + I(pni_pre / SD_PNI)
  f3 <- aki ~ cl + I(pni_pre / SD_PNI) + age + emale + asa + op_duration +
             cr + emop + open_
  f4 <- aki ~ cl + I(pni_post_first / SD_PNI) + age + emale + asa +
             op_duration + cr + emop + open_

  clt <- grep("^cl", names(coef(glm(f1, data = Dc, family = binomial))), value = TRUE)
  pn1 <- "I(pni_pre/SD_PNI)"; pn2 <- "I(pni_post_first/SD_PNI)"

  for (nm in c("M1","M2","M3","M4")) {
    g <- glm(get(paste0("f", substring(nm, 2))), data = Dc, family = binomial)
    tt <- clt
    if (nm == "M2" || nm == "M3") tt <- c(clt, pn1)
    if (nm == "M4") tt <- c(clt, pn2)
    v <- ors(g, tt)
    ## PNI 按 −1 SD 方向报告
    for (j in seq_along(tt)) {
      if (tt[j] %in% c(pn1, pn2)) {
        s <- summary(g)$coefficients; b <- s[tt[j], 1]; se <- s[tt[j], 2]
        v[j] <- sprintf("%.2f (%.2f-%.2f) p=%.4g  [per -1SD]", exp(-b/1),
                        exp(-b/1 + 1.96*se/1), exp(-b/1 - 1.96*se/1), s[tt[j], 4])
      }
    }
    for (j in seq_along(tt)) {
      lab <- tt[j]
      if (lab %in% c(pn1, pn2))
        lab <- if (lab == pn1) "PNI_术前(-1SD)" else "PNI_术后首值(-1SD)"
      RES[[length(RES) + 1]] <- data.frame(form = cd$form, ng = cd$ng, K = K,
        model = nm, n = nrow(Dc), events = sum(Dc$aki),
        term = lab, OR_CI = v[j], stringsAsFactors = FALSE)
    }
    cat(sprintf("    %s: %s\n", nm, paste(paste0(tt, "=", v), collapse="  ")))
  }
}

R <- do.call(rbind, RES)
write.csv(R, file.path(OUT, "lcmm_adjusted_ors.csv"), row.names = FALSE)
cat("\n已写出 lcmm_adjusted_ors.csv\n")
print(R, row.names = FALSE)
