;;   The Computer Language Benchmarks Game
;;   https://salsa.debian.org/benchmarksgame-team/benchmarksgame/
;;;
;;; contributed by Patrick Frankenberger
;;; modified by Juho Snellman 2005-11-18
;;;   * About 40% speedup on SBCL, 90% speedup on CMUCL
;;;   * Represent a body as a DEFSTRUCT with (:TYPE VECTOR DOUBLE-FLOAT), a
;;;     not as a structure that contains vectors
;;;   * Inline APPLYFORCES
;;;   * Replace (/ DT DISTANCE DISTANCE DISTANCE) with
;;;     (/ DT (* DISTANCE DISTANCE DISTANCE)), as is done in the other
;;;     implementations of this test.
;;;   * Add a couple of declarations
;;;   * Heavily rewritten for style (represent system as a list instead of
;;;     an array to make the nested iterations over it less clumsy, use
;;;     INCF/DECF where appropriate, break very long lines, etc)
;;; modified by Marko Kocic
;;;   * add optimization declarations

;; (declaim (optimize (speed 3)(safety 0)(space 0)(debug 0)))

(defpackage #:nbody
  (:use :cl :sb-ext :sb-c)
  (:export #:main
           #:nbody))

(in-package :nbody)

;; --------------- V3D -----------------------
(deftype v3d () '(simd-pack-256 double-float))

;; (when (not (zerop (sb-alien:extern-alien "avx2_supported" int))) "YES")

;; #.(when (member :sb-simd-pack-256 sb-impl:+internal-features+)
;;     (defun toto ()
;;       "yes"))

;; #.(when (not (member :sb-simd-pack-256 sb-impl:+internal-features+))
;;     (defun toto ()
;;       "no"))


(declaim (inline make-v3d))
(defun make-v3d (x y z)
  (declare (type double-float x y z))
  (%make-simd-pack-256-double x y z 0d0))

(sb-c:defknown (v3d+ v3d- v3d*) (v3d v3d) v3d
  (sb-c:movable sb-c:flushable sb-c:always-translatable)
  :overwrite-fndb-silently cl:t)

(in-package :sb-vm)

(define-vop (nbody::v3d+)
  (:translate nbody::v3d+)
  (:policy :fast-safe)
  (:args (x :scs (double-avx2-reg) :target r)
         (y :scs (double-avx2-reg)))
  (:arg-types simd-pack-256-double simd-pack-256-double)
  (:results (r :scs (double-avx2-reg)))
  (:result-types simd-pack-256-double)
  (:generator 4
              (inst vaddpd r x y)))

(define-vop (nbody::v3d*)
  (:translate nbody::v3d*)
  (:policy :fast-safe)
  (:args (x :scs (double-avx2-reg) :target r)
         (y :scs (double-avx2-reg)))
  (:arg-types simd-pack-256-double
              simd-pack-256-double)
  (:results (r :scs (double-avx2-reg)))
  (:result-types simd-pack-256-double)
  (:generator 4
              (inst vmulpd r x y)))

(define-vop (nbody::v3d-)
  (:translate nbody::v3d-)
  (:policy :fast-safe)
  (:args (x :scs (double-avx2-reg) :target r)
         (y :scs (double-avx2-reg)))
  (:arg-types simd-pack-256-double
              simd-pack-256-double)
  (:results (r :scs (double-avx2-reg) :from (:argument 0)))
  (:result-types simd-pack-256-double)
  (:generator 4
              (inst vsubpd r x y)))

(in-package :nbody)

(macrolet ((define-stub (name)
             `(defun ,name (x y)
                (,name x y))))
  (define-stub v3d+)
  (define-stub v3d*)
  (define-stub v3d-))

(defun v3d-norm^2 (x)
  (declare (type v3d x))
  (let ((x2 (v3d* x x)))
    (declare (type v3d x2))
    (+ (sb-vm::%simd-pack-256-double-item x2 0)
       (sb-vm::%simd-pack-256-double-item x2 1)
       (sb-vm::%simd-pack-256-double-item x2 2))))

(defun v3d-replicate (x)
  (declare (type double-float x))
  (make-v3d x x x))

(defconstant +days-per-year+ 365.24d0)
(defconstant +solar-mass+ (* 4d0 pi pi))

(defstruct body
  (position (make-v3d 0d0 0d0 0d0) :type v3d)
  (velocity (make-v3d 0d0 0d0 0d0) :type v3d)
  (mass 0d0 :type double-float))

(defun make-initial-solar-system ()
  (let ((sun
          (make-body :position (make-v3d 0.0d0 0.0d0 0.0d0)
                     :velocity (make-v3d 0.0d0 0.0d0 0.0d0)
                     :mass +solar-mass+))
        (jupiter
          (make-body :position (make-v3d 4.84143144246472090d0
                                         -1.16032004402742839d0
                                         -1.03622044471123109d-1)
                     :velocity (make-v3d (* 1.66007664274403694d-3 +days-per-year+)
                                         (* 7.69901118419740425d-3 +days-per-year+)
                                         (* -6.90460016972063023d-5  +days-per-year+))
                     :mass (* 9.54791938424326609d-4 +solar-mass+)))
        (saturn
          (make-body :position (make-v3d 8.34336671824457987d0
                                         4.12479856412430479d0
                                         -4.03523417114321381d-1)
                     :velocity (make-v3d (* -2.76742510726862411d-3 +days-per-year+)
                                         (* 4.99852801234917238d-3 +days-per-year+)
                                         (* 2.30417297573763929d-5 +days-per-year+))
                     :mass (* 2.85885980666130812d-4 +solar-mass+)))
        (uranus
          (make-body :position (make-v3d 1.28943695621391310d1
                                         -1.51111514016986312d1
                                         -2.23307578892655734d-1)
                     :velocity (make-v3d (* 2.96460137564761618d-03 +days-per-year+)
                                         (* 2.37847173959480950d-03 +days-per-year+)
                                         (* -2.96589568540237556d-05 +days-per-year+))
                     :mass (* 4.36624404335156298d-05 +solar-mass+)))
        (neptune
          (make-body :position (make-v3d 1.53796971148509165d+01
                                         -2.59193146099879641d+01
                                         1.79258772950371181d-01)
                     :velocity (make-v3d (* 2.68067772490389322d-03 +days-per-year+)
                                         (* 1.62824170038242295d-03 +days-per-year+)
                                         (* -9.51592254519715870d-05 +days-per-year+))
                     :mass (* 5.15138902046611451d-05 +solar-mass+))))
    (list sun jupiter saturn uranus neptune)))

(declaim (inline applyforces))
(defun applyforces (a b dt)
  (declare (type body a b)
           (type double-float dt))
  (let* ((dp (v3d- (body-position a) (body-position b)))
         (distance (sqrt (v3d-norm^2 dp)))
	       (mag (/ dt (* distance distance distance)))
         (dpmag (v3d* dp (make-v3d mag mag mag))))
    (setf (body-velocity a)
          (v3d- (body-velocity a)
                (v3d* dpmag
                      (v3d-replicate (body-mass b)))))
    (setf (body-velocity b)
          (v3d+ (body-velocity b)
                (v3d* dpmag
                      (v3d-replicate (body-mass a))))))
  nil)

(defun advance (system dt)
  (declare (double-float dt))
  (loop for (a . rest) on system do
    (dolist (b rest)
      (declare (type body b))
      (applyforces a b dt)))
  (dolist (b system)
    (declare (type body b))
    (setf (body-position b)
          (v3d+ (body-position b)
                (v3d* (v3d-replicate dt)
                      (body-velocity b)))))
  nil)

(defun energy (system)
  (let ((e 0.0d0))
    (declare (double-float e))
    (loop for (a . rest) on system do
      (incf e (* 0.5d0
                 (body-mass a)
                 (v3d-norm^2 (body-velocity a))))
      (dolist (b rest)
        (let* ((dp (v3d- (body-position a) (body-position b)))
               (dist (sqrt (v3d-norm^2 dp))))
          (decf e (/ (* (body-mass a) (body-mass b)) dist)))))
    e))

(defun offset-momentum (system)
  (let ((p (v3d-replicate 0d0)))
    (declare (type v3d p))
    (dolist (b system)
      (setf p (v3d+ p
                    (v3d* (body-velocity b)
                          (v3d-replicate (body-mass b))))))

    (setf (body-velocity (car system))
          (v3d* (v3d- (v3d-replicate 0d0)
                      p)
                (v3d-replicate (/ +solar-mass+))))
    nil))

(defun nbody (n)
  (declare (fixnum n))
  (let ((system (make-initial-solar-system)))
    (offset-momentum system)
    (format t "~,9F~%" (energy system))
    (dotimes (i n)
      (advance system 0.01d0))
    (format t "~,9F~%" (energy system))))

(defun main ()
  (let ((n (parse-integer (or (car (last #+sbcl sb-ext:*posix-argv*
                                         #+cmu  extensions:*command-line-strings*
					                               #+gcl  si::*command-args*)) "1"))))
    (nbody n)))
