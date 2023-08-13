#!/bin/bash
# ./scripts/adapt_tent/adapt_test_unsup.sh uchicago fixed
# ./scripts/adapt_tent/adapt_test_unsup.sh uchicago adapt
# ./scripts/adapt_tent/adapt_test_unsup.sh uchicago none

./scripts/adapt_tent/adapt_test_sup.sh uchicago fixed
./scripts/adapt_tent/adapt_test_sup.sh uchicago adapt
./scripts/adapt_tent/adapt_test_sup.sh uchicago none

# ./scripts/adapt_tent/adapt_test_sup.sh midrc fixed
# ./scripts/adapt_tent/adapt_test_sup.sh midrc adapt
# ./scripts/adapt_tent/adapt_test_sup.sh midrc none

# ./scripts/adapt_tent/adapt_test_sup.sh cohen fixed
# ./scripts/adapt_tent/adapt_test_sup.sh cohen adapt
# ./scripts/adapt_tent/adapt_test_sup.sh cohen none