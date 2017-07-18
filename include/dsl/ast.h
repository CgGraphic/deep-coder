#include <vector>
#include <string>
#include <cstdint>
#ifdef USE_OPTION

#define Optional(flag,x)  (x)
#define Optional(x)    (x)
#define OptExits(x)    (x)
#define OptValue(x)    (x.value())
#define OptValueOr(x,y) (x.value_or(y))

#if defined(_MSC_VER) && (_MSC_VER >1900)

#include <optional>

namespace std
{

	namespace experimental
	{

		template<class T>
		using optional = std::optional<T>;


	};
};
#else
#include <experimental/optional.hpp>
#endif


#else
namespace std
{

	namespace experimental
	{
		template<class T>
		using optional = std::pair<bool, T>;
	}
}

#define Optional(flag, x)  {(flag),(x)}
#define Optional(x) {true,(x)}

#define OptExists(x)    (x.first)
#define OptValue(x)    (x.second)
#define OptValueOr(x,y) (x.first?x.second:y)
#endif
#pragma once

namespace dsl {
    enum class Function {
        Head, Last, Take, Drop, Access, Minimum, Maximum, Reverse, Sort, Sum,
        Map, Filter, Count, ZipWith, Scanl1, ReadInt, ReadList
    };
    extern std::vector<Function> all_functions;

    enum class PredicateLambda {
        IsPositive = 0x40000000, IsNegative, IsEven, IsOdd
    };
    extern std::vector<PredicateLambda> all_predicate_lambdas;

    enum class TwoArgumentsLambda : uint32_t {
        Plus = 0x20000000, Minus, Multiply, Min, Max
    };
    extern std::vector<TwoArgumentsLambda> all_two_arguments_lambdas;

    enum class OneArgumentLambda : uint32_t {
        Plus1 = 0x10000000, Minus1, Multiply2, Divide2, MultiplyMinus1, Pow2, Multiply3, Divide3, Multiply4, Divide4
    };
    extern std::vector<OneArgumentLambda> all_one_argument_lambdas;

    using Variable = uint16_t;

    struct Argument {
        Argument(Variable variable);
        Argument(PredicateLambda lambda);
        Argument(TwoArgumentsLambda lambda);
        Argument(OneArgumentLambda lambda);

        std::experimental::optional<PredicateLambda> predicate() const;
        std::experimental::optional<TwoArgumentsLambda> two_arguments_lambda() const;
        std::experimental::optional<OneArgumentLambda> one_argument_lambda() const;
        std::experimental::optional<Variable> variable() const;

		uint32_t Val() const
		{
			return m_argument;
		}
    private:
        uint32_t m_argument;
    };

    struct Statement {
        Variable variable;
        Function function;
        std::vector<Argument> arguments;

        Statement(Variable variable, Function function, const std::vector<Argument> &arguments);
    };

    using Program = std::vector<Statement>;
}

