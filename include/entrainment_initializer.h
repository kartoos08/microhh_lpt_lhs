#ifndef ENTRAINMENT_INITIALIZER_H
#define ENTRAINMENT_INITIALIZER_H

class Input;
template<typename> class Grid;
template<typename> class Fields;

namespace Lpt
{
template<typename TF>
void apply_entrainment_initial_condition(Grid<TF>&, Fields<TF>&, Input&);
}

#endif
