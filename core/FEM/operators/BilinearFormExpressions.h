#ifndef __BILINEAR_FORM_EXPRESSIONS_H__
#define __BILINEAR_FORM_EXPRESSIONS_H__

#include "../MESH/Element.h"
using fdaPDE::core::MESH::Element;
#include "../utils/Symbols.h"

#define DEF_BILINEAR_FORM_EXPR_OPERATOR(OPERATOR, FUNCTOR)		\
  template <typename E1, typename E2>					\
  BilinearFormBinOp<E1, E2, FUNCTOR >					\
  OPERATOR(const BilinearFormExpr<E1>& op1,				\
	   const BilinearFormExpr<E2>& op2) {				\
    return BilinearFormBinOp<E1, E2, FUNCTOR >				\
      {op1.get(), op2.get(), FUNCTOR()};				\
  }									\
  									
template <typename E> struct BilinearFormExpr{
  // call integration method on base type
  template <unsigned int N, int M, unsigned int ORDER, typename B>
  double integrate(const B& b, const Element<ORDER, N>& e, int i , int j, const SVector<M>& qp) const{
    return static_cast<const E&>(*this).integrate(b, e, i, j, qp);
  }
  
  // get underyling type composing the expression node
  const E& get() const { return static_cast<const E&>(*this); }

  // computes if the expression refers to a symmetric bilinear form (this can be used to activate useful optimizations)
  constexpr bool isSymmetric() const{
    return static_cast<const E&>(*this).isSymmetric();
  };

};

// a generic binary operation node
template <typename OP1, typename OP2, typename BinaryOperation>
class BilinearFormBinOp : public BilinearFormExpr<BilinearFormBinOp<OP1, OP2, BinaryOperation>> {
private:
  typename std::remove_reference<OP1>::type op1_;   // first  operand
  typename std::remove_reference<OP2>::type op2_;   // second operand
  BinaryOperation f_;                               // operation to apply

public:
  // constructor
  BilinearFormBinOp(const OP1& op1, const OP2& op2, BinaryOperation f) : op1_(op1), op2_(op2), f_(f) { };

  // integrate method. Apply the functor f_ to the result of integrate() applied to both operands.
  template <unsigned int N, int M, unsigned int ORDER, typename B>
  double integrate(const B& b, const Element<ORDER, N>& e, int i , int j, const SVector<M>& qp) const{
    return f_(op1_.integrate(b, e, i, j, qp), op2_.integrate(b, e, i, j, qp));
  }

  // computes if the node refers to a symmetric bilinear form (both operands are symmetric, ok for linear operations)
  constexpr bool isSymmetric() const{ return op1_.isSymmetric() && op2_.isSymmetric(); }
};

// node representing a single scalar in an expression
class BilinearFormScalar : public BilinearFormExpr<BilinearFormScalar> {
private:
  double value_;
public:
  constexpr bool isSymmetric() const { return true;} // constant never break symmetry of the operator

  // constructor
  BilinearFormScalar(double value) : value_(value) { }
  
  // integrate method. return the stored value
  template <unsigned int N, int M, unsigned int ORDER, typename B>
  double integrate(const B& b, const Element<ORDER, N>& e, int i , int j, const SVector<M>& qp) const{
    return value_;
  }
};

// allow scalar*operator expressions
template <typename E>                                                 
BilinearFormBinOp<BilinearFormScalar, E, std::multiplies<> >
operator*(double op1, const BilinearFormExpr<E> &op2) {
  return BilinearFormBinOp<BilinearFormScalar, E, std::multiplies<> >
    (BilinearFormScalar(op1), op2.get(), std::multiplies<>());
}  

DEF_BILINEAR_FORM_EXPR_OPERATOR(operator+, std::plus<>)
DEF_BILINEAR_FORM_EXPR_OPERATOR(operator-, std::minus<>)

#endif // __BILINEAR_FORM_EXPRESSIONS_H__